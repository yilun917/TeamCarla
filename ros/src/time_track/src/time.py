#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32, Bool

def loop():
    global my_pub
    global initilizing
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        my_pub.publish(initilizing)
        if(initilizing.data):
            initilizing.data = False
        rate.sleep()

def time_cb(msg):
    # if there is any message in topic "traffic_waypoint" and is first time
    # pass a False value to let the control node halt for 3 s
    global initilizing
    global first
    if(first):
        initilizing.data = True
        first = False

rospy.init_node("time_node")
initilizing = Bool()
initilizing.data = False
first = True
my_pub = rospy.Publisher("/time_track", Bool, queue_size=1)
my_sub = rospy.Subscriber("/traffic_waypoint", Int32, time_cb)
loop()




'''
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
'''
   


  
