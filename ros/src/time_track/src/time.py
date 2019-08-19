#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32, Bool


class TimeTracker():
    def __init__(self):
        self.check = Bool()
        self.check.data = True
        self.first = True
        rospy.init_node("time_node")
        self.my_sub = rospy.Subscriber("/traffic_waypoint", Int32,  self.time_cb)
        self.my_pub = rospy.Publisher("/time_track", Bool)
        self.loop()

    def time_cb(self, msg):
        # if there is any message in topic "traffic_waypoint" and is first time
        # pass a False value to let the control node halt for 3 s
        if(self.first):
            self.check.data = False
            self.first = False

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.my_pub.publish(self.check)
            self.check.data = True
            rate.sleep()

   


  
