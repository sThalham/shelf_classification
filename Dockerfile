FROM tensorflow/tensorflow:2.6.0-gpu
#FROM ros:melodic
ENV DEBIAN_FRONTEND noninteractive

# install ros package
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get update && apt-get install -y \
	ros-melodic-ros-core \
      	ros-melodic-libuvc-camera \
      	ros-melodic-image-view \
      	ros-melodic-cv-bridge \
      	ros-melodic-cv-camera \
      	ros-melodic-actionlib \
	ros-melodic-catkin \
	ros-melodic-ros-numpy \
	python-rosdep \
 	python-rosinstall \ 
	python-rosinstall-generator \ 
	python-wstool \
	build-essential \
	python-vcstools \
	python-catkin-tools \
	vim && \
    rm -rf /var/lib/apt/lists/*

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-dev \
     python3-numpy \
     python3-pip \
     && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
    rosdep update
# catkin tools
RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python-catkin-tools \
     && rm -rf /var/lib/apt/lists/*

# install python packages
RUN pip install --upgrade pip setuptools
RUN pip3 install --upgrade rospkg catkin_pkg opencv-contrib-python empy
# for ros environments
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /catkin_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so; catkin_make'
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

#ENTRYPOINT ["/ros_entrypoint.sh"]
#CMD ["bash"]

