#!/bin/bash

sudo docker build --no-cache -t ros_container .
sudo docker run --network=host --name=ros_container -t -d -v ~/shelf_classifier:/shelf_classifier --env ROS_MASTER_URI=http://rent-a-pc:11311 ros_container


