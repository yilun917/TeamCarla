#!/usr/bin/env python
import rospy
import tensorflow as tf
import cv2
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        # Frozen inference graph files. NOTE: change the path to where you saved the models.
        SSD_GRAPH_FILE = '../../../data/frozen_inference_graph.pb'

        self.detection_graph = self.load_graph(SSD_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Add the growth opption to the config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default() as graph:
            graph_init_op = tf.global_variables_initializer()
        
        with self.detection_graph.as_default():
            self.sess = tf.Session(config=config, graph=self.detection_graph)

        # Create a tf session
        self.sess = tf.Session(config=config, graph=self.detection_graph)
        self.sess.run(graph_init_op)

    def __del__(self):
        # Close tf the session
        self.sess.close() 
        rospy.loginfo("tl_classifier::TF SESS CLOSED")
               

    """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
    """
    def get_classification(self, image):

        confidence_cutoff = 0.6
        detect_types = [10.]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #start_time = time.time()
        # Crop the image to our ROI - this will help filter out other lanes and the ground, also make image smaller for faster processing
        #crop_img = image.copy()
        #h = crop_img.shape[0]
        #w = crop_img.shape[1]

        # High center image
        #y = 0             # Adjust top start
        #x = int(h * 0.25) # Adjust side to side start
        #h = int(h * 0.80) # Adjust window height
        #w = int(w * 0.55) # Adjust window width

        # Low center image
        #y = int(h * 0.30)  # Adjust top start
        #x = int(h * 0.25) # Adjust side to side start
        #h = int(h * 0.90) # Adjust window height
        #w = int(w * 0.65) # Adjust window width

        #crop_img = image[y: y + h, x: x + w].copy()
        lights = self.find_objects(image, confidence_cutoff, detect_types)

        labels = []

        if len(lights) == 0:
            rospy.loginfo("tl_classifier::No lights found")
            return TrafficLight.UNKNOWN

        # Loop through the traffic light detections and add the detected colors to the llabels list
        for i in range(len(lights)):
            label = self.estimate_label(lights[i])
            if label == [1, 0, 0]:  # RED
                labels.append(0)
            elif label == [0, 1, 0]: # YELLOW
                labels.append(1)
            elif label == [0, 0, 1]: # GREEN
                labels.append(2)

        # Look through the list and return the most dangerous color found
        if 0 in labels:  # RED
            rospy.loginfo("tl_classifier::Red light found")
            return TrafficLight.RED
        elif 1 in labels: # YELLOW
            rospy.loginfo("tl_classifier::Yellow light found")
            return TrafficLight.YELLOW
        elif 2 in labels: # GREEN
            rospy.loginfo("tl_classifier::Green light found")
            return TrafficLight.GREEN

        rospy.loginfo("tl_classifier::Light found - COLOR UNDETERMINED")
        return TrafficLight.UNKNOWN


    ### Color Classification Functions ###
    """This function should take in an RGB image and return a new, standardized version
        Args:
            image (cv::Mat): image to be standardized
        Returns:
            int: standardized image
    """
    def standardize_input(self, image):
        # Resize image and pre-process so that all images are the same size
        standard_im = cv2.resize(np.copy(image), (32, 32))

        return standard_im


    """Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
        Args:
            rgb_image (cv::Mat): image to be manipulated
        Returns:
            array: array of brightness levels
    """
    def create_feature(self, rgb_image):
        # Crop and split the image into three sections
        top = rgb_image[2:10 , 4:26, :].copy()
        top_hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
        middle = rgb_image[11:19 , 4:26, :].copy()
        middle_hsv = cv2.cvtColor(middle, cv2.COLOR_RGB2HSV)
        bottom = rgb_image[20:29 , 4:26, :].copy()
        bottom_hsv = cv2.cvtColor(bottom, cv2.COLOR_RGB2HSV)

        # Mask for Red
        lower_red_1 = np.array([5,5,150])
        upper_red_1 = np.array([12,255,255])
        lower_red_2 = np.array([150,28,120])
        upper_red_2 = np.array([179,255,255])
        top_mask_1 = cv2.inRange(top_hsv, lower_red_1, upper_red_1)
        top_mask_2 = cv2.inRange(top_hsv, lower_red_2, upper_red_2)
        top_mask = cv2.addWeighted(top_mask_1, 1, top_mask_2, 1, 0)

        # Mask for Yellow
        lower_yel = np.array([15,10,78])
        upper_yel = np.array([36,255,255])
        middle_mask = cv2.inRange(middle_hsv, lower_yel, upper_yel)

        # Mask for Green
        lower_grn = np.array([50,13,100]) #[50,20,150]
        upper_grn = np.array([100,255,255])
        bottom_mask = cv2.inRange(bottom_hsv, lower_grn, upper_grn)

        # Find the brightness of each section
        top_brightness = np.sum(top_mask)
        middle_brightness = np.sum(middle_mask)
        bottom_brightness = np.sum(bottom_mask)

        # Find the area of each section
        top_area = top.shape[0] * top.shape[1]
        middle_area = middle.shape[0] * middle.shape[1]
        bottom_area = bottom.shape[0] * bottom.shape[1]

        # Compute averages
        top_avg = top_brightness / top_area
        middle_avg = middle_brightness / middle_area
        bottom_avg = bottom_brightness / bottom_area

        # Return the average brightness of each section
        feature = [top_avg, middle_avg, bottom_avg]

        return feature


    """Analyze image using brightness feature creation and output a one-hot encoded label
        Args:
            rgb_image (cv::Mat): traffic lught image to detect current color of
        Returns:
            array: one hot encoded array
    """
    def estimate_label(self, rgb_image):
        # Standardize the image
        std_image = self.standardize_input(rgb_image)

        # Extract feature(s) from the RGB image and use those features to
        # classify the image and output a one-hot encoded label
        feature_vals = self.create_feature(std_image)
        max_ind = feature_vals.index(max(feature_vals))

        predicted_label = [1, 0, 0]

        if max_ind == 1:
            predicted_label = [0, 1, 0]

        if max_ind == 2:
            predicted_label = [0, 0, 1]

        #print("red: ", feature_vals[0],"yellow: ", feature_vals[1], "green: ", feature_vals[2])

        return predicted_label



    ### Traffic Light Classification Functions ###
    """Return only boxes that are high enough in score and are of the correct type
        Args:
            min_score (double): confidence the detection must meed
            detect_types (array): types of objects to be found by detection
            boxes  (array): bounding boxes
            scores (array): scores for detections
            classes (array): detection classes
        Returns:
            (array): bounding boxes
            (array): scores for detections
            (array): detection classes
    """
    def filter_boxes(self, min_score, detect_types, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score and classes[i] in detect_types:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes


    """Return image coordinates of bounding box
        Args:
            boxes  (array): bounding boxes
            height (int): height of image
            width (int): width of image
        Returns:
            (array): bounding boxes
            (array): scores for detections
            (array): detection classes
    """
    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords


    """Load the detection graph
    Args:
        graph_file  (file): graph file
    Returns:
        (graph): graph
    """
    def load_graph(self, graph_file):
        # Loads frozen inference graph
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    """Find objects in image
    Args:
        image (cv::Mat): image to detect objects in
        confidence_cutoff (double): detection threshold
        detect_types (array): types of objects to detect
    Returns:
        (array): detected objects of specified type(s)
    """
    def find_objects(self, image, confidence_cutoff, detect_types):
        # Convert to numpy array for detection processing
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        
        # Process the image
        #start_time = time.time()
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_np})
        #end_time_detect = time.time()
        #print "tf detect time: ", end_time_detect-start_time

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, detect_types, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        #width, height = image.size
        #box_coords = to_image_coords(boxes, height, width)
        size = image.shape
        box_coords = self.to_image_coords(boxes, size[0], size[1])

        found_objects = []

        # Loop through the bounding boxes, convert to a cropped image at the bounding box and add to the list to return
        for i in range(len(box_coords)):
            crop_img = image.copy()
            crop_img = crop_img[int(box_coords[i][0]):int(box_coords[i][2]), int(box_coords[i][1]):int(box_coords[i][3])]
            found_objects.append(crop_img)

        end_time_classify = time.time()
        #print "tf classification time: ", end_time_classify-end_time_detect
        #print "tf TOTAL TIME: ", end_time_classify-start_time

        return found_objects
