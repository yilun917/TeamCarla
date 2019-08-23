#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, String
from styx_msgs.msg import TrafficLight

global my_pub


def loop():
    rate = rospy.Rate(50)
    
    while not rospy.is_shutdown():
        rate.sleep()


def traffic_cb(msg):
    # if there is any message in topic "tl_color" let other modules know it is publishing
    my_pub.publish(True)


rospy.init_node("time_node")

my_pub = rospy.Publisher("/time_track", Bool, queue_size=1)
my_sub = rospy.Subscriber("/tl_color", String, traffic_cb)

loop()