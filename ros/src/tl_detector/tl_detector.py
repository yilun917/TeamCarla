#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import time
from std_msgs.msg import String
import numpy as np


STATE_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.has_image = False
        self.lights = []
        self.last_light_state = TrafficLight.UNKNOWN
        self.immediate_light_state = TrafficLight.UNKNOWN
        self.immediate_light_state_count = 0
        self.light_classifier_throttle_time = rospy.get_rostime()

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        self.test_pub = rospy.Publisher('/dummy_image', Image, queue_size=1)
        self.light_pub = rospy.Publisher('/tl_filtered', String, queue_size=1)


        config_string = rospy.get_param("/traffic_light_config")
        # config also contains an "is_site" bool variable to check if the car is running in simulator or site
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.light_classifier.find_objects(np.zeros((2,2,3), np.uint8), 0.5, [10.]) # Hit the classifier to get TF through its init
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.main_loop()

        #rospy.spin()

    def main_loop(self):
        while not rospy.is_shutdown():
            if(self.has_image):
                light_wp, state = self.process_traffic_lights()

                if self.state != state:
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))

                self.state_count += 1

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)  # store in a tree structure for easier access

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        self.test_pub.publish(self.camera_image)


    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # find the x, y position in the waypoints in the KDTree
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # just for testing
        #return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        #Get classification
        # Throttle the incoming requests
        #if self.light_classifier_throttle_time + rospy.Duration(0.05) < rospy.get_rostime():
            #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            #self.last_light_state = self.light_classifier.get_classification(cv_image)
            #self.light_classifier_throttle_time = rospy.get_rostime()
            #return self.last_light_state


        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        self.immediate_light_state = self.light_classifier.get_classification(cv_image)
        self.light_classifier_throttle_time = rospy.get_rostime()

        # Filter out noise from the classifier
        if (self.immediate_light_state != self.last_light_state):
            self.immediate_light_state_count += 1
            if (self.immediate_light_state_count > 6):
                if (self.last_light_state == 1 and self.immediate_light_state != 0):
                    self.last_light_state =  self.last_light_state
                else:
                    self.last_light_state = self.immediate_light_state
                self.immediate_light_state_count = 0

        self.light_pub.publish(str(self.last_light_state))
        return self.last_light_state

        #else:
        #    return self.last_light_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            car_wp_idx = self.get_closest_waypoint(x, y)

            #TODO find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
            # iterate through the traffic lights to find the closest
        for i, light in enumerate(self.lights):
            # Get stop line waypoint index
            line = stop_line_positions[i]
            temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
            # Find closest stop line waypoint idx
            d = temp_wp_idx - car_wp_idx
            # linear check to find the closest
            if d >= 0 and d < diff:
                diff = d
                closest_light = light
                line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
