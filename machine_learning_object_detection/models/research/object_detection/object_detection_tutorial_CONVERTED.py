#!/usr/bin/env python

import os
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from scipy.stats import norm
from scipy.interpolate import interp1d

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

DEBUG = 0

NUM_SECTION = 8
MAX_X = 1280
MAX_Y = 720
CENTER_X = int(MAX_X / 2)
SECTION_HEIGHT = int(MAX_Y / NUM_SECTION)
H_SECTION = 10
X_DIVISION = 32
CROP_THRESH = 0.5
assert(int(SECTION_HEIGHT) == SECTION_HEIGHT)

x_axis = np.arange(-10, 10, 0.001)
out = (1 - norm.pdf(x_axis, 0 , 3)[::int(len(x_axis)/MAX_X)] * 3)
normDist = out[int((len(out) - MAX_X)/2) : int((len(out) - MAX_X)/2) + 1280]

if DEBUG == 3:
    plt.plot(normDist)
    plt.show()

cap = cv2.VideoCapture("/Users/xuanchen/Desktop/test2.mp4")

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# ## Loader
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# # Detection
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)


# Check the model's input signature, it expects a batch of 3-color images of type uint8: 


print(detection_model.inputs)


detection_model.output_dtypes

detection_model.output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inferenceq
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


while True :
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  ret, image_np = cap.read()

  frame = image_np

  ######################################
  ########## OBJECT DETECTION ##########
  ######################################

  # Actual detection.
  #image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
  image_np = cv2.resize(image_np, (MAX_X, MAX_Y))

  output_dict = run_inference_for_single_image(detection_model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
  
  boxes, classes, scores = output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']

  for i, b in enumerate (boxes) :
      # 1: person
      if classes[i] == 1 :
        # accuracy
          if scores[i] > 0.5 :

              mid_x = (boxes[i][3] + boxes[i][1]) / 2
              mid_y = (boxes[i][2] + boxes[i][0]) / 2

              apx_distance = round( (1 - (boxes[i][3] - boxes[i][1])) ** 4, 1)

              cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 1280) - 25, int(mid_y * 720) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

              if apx_distance <= 0.6 :
                 if mid_x > 0.3 and mid_x < 0.7 :
                    cv2.putText(image_np, 'WARNING!!'.format(apx_distance), (int(mid_x * 1280) - 50, int(mid_y * 720) - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    
  ####################################
  ########## PATH DETECTION ##########
  ####################################

  #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
  frame = cv2.resize(frame,(MAX_X,MAX_Y))


  frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  frame_canny = cv2.Canny(frame_gray, 80, 250, apertureSize = 3)

  sections = list()
  for i in range(NUM_SECTION) :
    sections.append(frame_canny[int(SECTION_HEIGHT * i) : int(SECTION_HEIGHT * (i + 1))])

  sections.reverse()
  i = 0
  sections_rgb = list()

  bound_left = list()
  bound_right = list()

  for section in sections :

    # find histogram
    hist = np.sum(section, axis=0)
    #if i < 3 :
    hist = hist * normDist

    h_mean = np.mean(hist)
    h_max = np.max(hist)
    hist[hist < h_max/2] = 0

    section_rgb = cv2.cvtColor(section, cv2.COLOR_GRAY2RGB) # back to rbg for display purpose
    if DEBUG == 2:
        for index in range(MAX_X - 1): # direct
            cv2.line(section_rgb,(index , SECTION_HEIGHT - int((hist[index]/h_max)*SECTION_HEIGHT)),(index + 1 , SECTION_HEIGHT - int((hist[index + 1]/h_max)*SECTION_HEIGHT)),(0,0,255),1)
        

    result = np.reshape(hist, (-1, X_DIVISION)).sum(axis=-1)
    
    if DEBUG == 2 :
        j = 0
        for point in result :
            cv2.circle(section_rgb, (int(X_DIVISION/2 + j * X_DIVISION), SECTION_HEIGHT - int(point/np.max(result)*SECTION_HEIGHT)), 5, (0,255,0), thickness=-1 )
            j = j + 1
        #for index in range(x_average): # x average
        
    left_half = np.asarray(result[:len(result)//2])
    right_half = np.asarray(result[len(result)//2:])
    assert(len(left_half) == len(right_half))
    left_max = np.argmax(left_half)
    right_half = right_half[::-1]
    right_max = np.argmax(right_half)

    #print(left_max, right_max)
        
    ###CHANCHE TO SLOP DECISION
    left_half_s1 = left_half[1::1].tolist()
    left_half_s1.append(left_half[left_max])
    right_half_s1 = right_half[1::1].tolist()
    right_half_s1.append(right_half[right_max])

    left_diff = left_half - left_half_s1
    right_diff = right_half - right_half_s1
    #right_diff = right_diff[::-1]
    left_bound = np.argmax(left_diff)
    right_bound = MAX_X//X_DIVISION - np.argmax(right_diff) - 1
  
    if DEBUG == 1 :
        cv2.line(section_rgb, (int(left_bound * X_DIVISION), 0), (int(left_bound * X_DIVISION), SECTION_HEIGHT), (255, 0, 0), 5)
        cv2.line(section_rgb, (int(right_bound * X_DIVISION), 0), (int(right_bound * X_DIVISION), SECTION_HEIGHT), (255, 0, 0), 5)

    bound_left.append(left_bound * X_DIVISION)
    bound_right.append(right_bound * X_DIVISION)
    sections_rgb.append(section_rgb)
    i = i + 1

  sections_rgb.reverse()
  numpy_vertical = np.vstack(sections_rgb)
  #left
  y = list()
  y.append(int(0))
  y.append(int(SECTION_HEIGHT/2))

  for x in bound_left:
        y.append(int(y[-1] + SECTION_HEIGHT))

  y[-1] = int(y[-2] + SECTION_HEIGHT/2)
  bound_left.insert(0, bound_left[0])
  bound_left.append(bound_left[-1])
    
  y_new = np.linspace(0, 719, num=720, endpoint=True)
  f = interp1d(y, bound_left, kind='cubic')
  x_new = f(y_new)
  x_new = np.round(x_new)
  x_new[x_new < 0] = 0
  x_new[x_new > MAX_X] = MAX_X

  pts = np.matrix((np.flip(x_new, 0),y_new))
  pts = np.asarray(pts.transpose())
  cv2.polylines(image_np, np.int32([pts[300:]]), False, (0, 255, 0), 5)

  #right
  bound_right.insert(0, bound_right[0])
  bound_right.append(bound_right[-1])

  f = interp1d(y, bound_right, kind = 'cubic')
  x_new = f(y_new)
  x_new = np.round(x_new)
  x_new[x_new < 0] = 0
  x_new[x_new > MAX_X] = MAX_X

  pts = np.matrix((np.flip(x_new, 0),y_new))
  pts = np.asarray(pts.transpose())

  cv2.polylines(image_np, np.int32([pts[300:]]), False, (0, 255, 0), 5)

  #added = cv2.addWeighted(frame, 1, numpy_vertical, 1, 0)

  #############################
  ########## DISPLAY ##########
  #############################

  #final = cv2.addWeighted(added, 1, image_np, 1, 0)

  cv2.imshow('object_detection', image_np)
  
  if cv2.waitKey(25) & 0xFF == ord('q') :
    cv2.destroyAllWindows()
    break



