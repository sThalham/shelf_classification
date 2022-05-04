#!/bin/bash

sudo docker build --no-cache -t ros_container .
sudo docker run --network=host --name=ros_container -t -d -v ~/shelf_classification:/shelf_classifier --env ROS_MASTER_URI=http://192.168.0.235:11311 ros_container


