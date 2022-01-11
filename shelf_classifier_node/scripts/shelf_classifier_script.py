#!/usr/bin/env python

import sys
import os
import inspect

import rospy
from std_msgs.msg import String, Header, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from shelf_classifier_node.srv import returnClass

import argparse
import math
import copy

import tensorflow.keras as K
import numpy as np
import os
import cv2
import ros_numpy


class ContinuousClassification:
    def __init__(self, model, classes, topic):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._model = model
        self._classes = classes
        
        self.bridge = CvBridge()
        self.topic = topic
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.cls_pub = rospy.Publisher("/shelf_classifier/class", String, queue_size=10)
        self.header_pub = rospy.Publisher("/shelf_classifier/header", Header, queue_size=10)

    def callback(self, data):
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))
        #self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")        
        img_rgb = cv2.resize(self._msg, (224, 168))
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        class_result = self._model.predict(img_rgb[np.newaxis, ...])
        prediction = np.argmax(class_result)
        classification = str(self._classes[prediction])


        self.publish_class(classification)

    def publish_class(self, classification):
        msg = String()
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = self.time
        header.seq = self.seq
        msg.data = classification
        self.cls_pub.publish(msg)
        self.header_pub.publish(header)


class ServiceClassification:
    def __init__(self, model, classes, topic, service_name):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._model = model
        self._classes = classes

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.topic = topic
        self.srv = rospy.Service(service_name, returnClass, self.callback)
        self.image_sub = rospy.Subscriber(self.topic, Image, self.image_callback)
        
    def image_callback(self, data):
        self.image = data

    def callback(self, req):
        #print(data)
        rospy.wait_for_message(self.topic, Image)
        data = self.image
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = np.frombuffer(data.data, dtype=np.uint8).reshape((data.height, data.width, -1))
        #self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")        
        img_rgb = cv2.resize(self._msg, (224, 168))
        img_rgb = cv2.cvtColor(self._msg, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        class_result = self._model.predict(img_rgb[np.newaxis, ...])
        prediction = np.argmax(class_result)
        confidence = class_result[0][prediction]
        classification = str(self._classes[prediction])
        print('Classified image as ', classification, ' with confidence: ', confidence)
        msg = self.fill_msg(classification, confidence)
        return msg

    def fill_msg(self, classification, confidence):
        header = Header()
        cls = String()
        cnf = Float32()
        header.frame_id = self.frame_id
        header.stamp = self.time
        header.seq = self.seq
        cls.data = classification
        cnf.data = confidence
        
        return [header, cls, cnf]


if __name__ == '__main__':

    # ROS params
    repository_path = ''			# path to github repo
    model_path = ''   				# path to trained model
    model_type = 'classifier' 			# [classify, segmenter]
    msg_topic = '/camera/rgb/image_color'	# ROS-topic
    detection_threshold = 0.5			# only used with 'segmenter' for as threshold for class-agnostic image manipulation
    node_type = 'continuous' 			# [continuous, service]
    service_name = 'getClasses'			# only used for service

    try:
        repository_path = rospy.get_param('/shelf_classifier/meshes_path')
    except KeyError:
        print("please set path to repository example:/home/desired/path/to/github_repo/")
    try:
        model_path = rospy.get_param('/shelf_classifier/model_path')
    except KeyError:
        print("please set path to model! example:/home/desired/path/to/resnet_xy.h5")

    if rospy.has_param('/shelf_classifier/model_type'): 
        model_type = rospy.get_param("/shelf_classifier/model_type")
        print(model_type)
        if model_type == 'classifier':
            pass
        elif model_type == 'segmenter':
            sys.exit('This model_type is not supported yet')
        else:
            sys.exit('unsupported model_type')
        print('Setting detection threshold not supported yet')
    if rospy.has_param('/shelf_classifier/image_topic'):
        msg_topic = rospy.get_param("/shelf_classifier/image_topic")
        print("Subscribing to msg topic: ", msg_topic)
    if rospy.has_param('/shelf_classifier/detection_threshold'):
        detection_threshold = rospy.get_param("/shelf_classifier/detection_threshold")
        print('Detection threshold set to: ', detection_threshold)
        print('Detection threshold usage only supported when using as segmenter')
    if rospy.has_param('/shelf_classifier/node_type'):
        node_type = rospy.get_param("/shelf_classifier/node_type")
        if node_type not in ['continuous', 'service']:
            sys.exit('Make sure node_type is one of [continuous, service]')
        print("node_type set to: ", node_type)
    if rospy.has_param('/shelf_classifier/service_call'):
        service_name = rospy.get_param("/shelf_classifier/service_call")
        print("service call set to: ", service_name)
        print('service_name only used with node_type service')
    
    classifier = K.models.load_model(model_path)
    classes = ["bucket", "hanging", "standing"]

    try:
        if rospy.get_param('/shelf_classifier/node_type') == 'continuous':
            print("node type set to continuous")
            shelf_classification = ContinuousClassification(classifier, classes, msg_topic)
        elif rospy.get_param('/shelf_classifier/node_type') == 'service':
            print("node type set to service")
            shelf_classification = ServiceClassification(classifier, classes, msg_topic, service_name)
    except KeyError:
        print("node_type should either be continuous or service.")

    rospy.init_node('shelf_classifier', anonymous=True)

    rospy.spin()






