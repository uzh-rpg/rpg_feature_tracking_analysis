#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
from cv_bridge import CvBridge

from tracker_utils import q_to_R


class Bag2Images:
    def __init__(self, path_to_bag, topic):
        self.path_to_bag = path_to_bag

        self.times = []
        self.images = []

        self.bridge = CvBridge()

        with rosbag.Bag(path_to_bag) as bag:
            
            topics = bag.get_type_and_topic_info().topics
            if topic not in topics:
                raise ValueError("The topic with name %s was not found in bag %s" % (topic, path_to_bag))

            for topic, msg, t in bag.read_messages(topics=[topic]):
                time = msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs
                self.times.append(time)

                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.images.append(img)


class Bag2CameraInfo:
    def __init__(self, path_to_bag, topic):
        self.K = None
        self.img_size = None

        with rosbag.Bag(path_to_bag) as bag:

            topics = bag.get_type_and_topic_info().topics
            if topic not in topics:
                raise ValueError("The topic with name %s was not found in bag %s" % (topic, path_to_bag))

            for topic, msg, t in bag.read_messages(topics=[topic]):
                self.K = np.array(msg.K).reshape((3,3))
                self.img_size = (msg.width, msg.height)
                break


class Bag2DepthMap:
    def __init__(self, path_to_bag, topic):
        self.times = []
        self.depth_maps = []

        self.bridge = CvBridge()

        with rosbag.Bag(path_to_bag) as bag:

            topics = bag.get_type_and_topic_info().topics
            if topic not in topics:
                raise ValueError("The topic with name %s was not found in bag %s" % (topic, path_to_bag))

            for topic, msg, t in bag.read_messages(topics=[topic]):
                time = msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs
                self.times.append(time)

                img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.depth_maps.append(img)


class Bag2Trajectory:
    def __init__(self, path_to_bag, topic):
        self.times = []
        self.poses = []
        self.quaternions = []

        self.bridge = CvBridge()

        with rosbag.Bag(path_to_bag) as bag:

            topics = bag.get_type_and_topic_info().topics
            if topic not in topics:
                raise ValueError("The topic with name %s was not found in bag %s" % (topic, path_to_bag))

            for topic, msg, t in bag.read_messages(topics=[topic]):
                time = msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs
                self.times.append(time)

                q = msg.pose.orientation
                pos = msg.pose.position
                quaternion = np.array([q.w, q.x, q.y, q.z])
                t = np.array([[pos.x], [pos.y], [pos.z]])

                R = q_to_R(quaternion)[0]

                transform = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])

                self.quaternions.append(quaternion)
                self.poses.append(transform)