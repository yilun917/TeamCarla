#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32, Bool


class TimeTracker():
    def __init__(self):
        rospy.init_node("time_node")
        self.check = Bool()
        self.check.data = True
        self.first = True
        self.my_pub = rospy.Publisher("/time_track", Bool)
        self.my_sub = rospy.Subscriber("/traffic_waypoint", Int32, self.time_cb)
        rospy.spin()

    def time_cb(self, msg):
        # if there is any message in topic "traffic_waypoint" and is first time
        # pass a False value to let the control node halt for 3 s
        if(self.first):
            print('initilizing')
            self.check.data = False
            self.first = False
            self.my_pub.publish(self.check)
            self.check.data = True


if __name__ == 'main':
    print("///////////////////////////////////////////////////////////////////")
    TimeTracker()

   


  
