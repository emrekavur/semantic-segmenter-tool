#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
This script is written for semantic segmentation for the input image(s)

This script uses pre-trained DeepLabv3+ model which is downloaded from:
http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz

Usage of the script: python segmenter.py "ImageName.jpg"

The function needs;
-Python > 3.6
-Numpy
-Scipy
-Tensorflow

You may install requirements with executing the following lines one by one:
pip install numpy
pip install scipy
pip install Pillow
pip install tensorflow

Credits:
1) DeepLab: https://github.com/tensorflow/models/tree/master/research/deeplab
This code is written with help of sample codes in project page.

2) Tensorflow: Large-Scale Machine Learning on Heterogeneous Distributed Systems
"""

import os
import sys # To call function with arguments
import tarfile # To open pre-trained data file in tar archieve
import numpy as np # To create Numpy arrays
import scipy.io # To save segmentation map as MATLAB .mat file
from PIL import Image # To make image operations
import tensorflow as tf # To run the model

def run_segmentation(s):
    image = Image.open(s) # Read image in argument
	
	#Define Tensor paameters
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
    
    tarball_path="deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz" # Pre-trained model file
    graph = tf.Graph() # Create a new TensorFlow graph
    graph_def = None
	
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
        if FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
            file_handle = tar_file.extractfile(tar_info)
            graph_def = tf.GraphDef.FromString(file_handle.read())
            break    
    tar_file.close()    
    
    if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')
    with graph.as_default():
        tf.import_graph_def(graph_def, name='') # Import graph of pre-trained network
		
    sess = tf.Session(graph=graph)    # Start a new TensorFlow session 
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height) 
    target_size = (int(resize_ratio * width), int(resize_ratio * height)) 
    resized_image = image.convert('RGB').resize(target_size, Image.NEAREST) # Resize image to the target size
	
    batch_seg_map = sess.run( # Run TensorFlow session
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]}) # Feed the network with the image in argument
		
    seg_map = batch_seg_map[0] #Grap segmentation map after finishing TensorFlow session
    seg_map_img = Image.fromarray(seg_map.astype(np.uint8)) # Convert segmentation map to image
    seg_map_img =seg_map_img.resize((width,height), Image.NEAREST) # Resize image to its original size
    seg_map_img_array = np.array(seg_map_img) # Convert image to Numpy array again (to save .mat file)
    scipy.io.savemat('seg_map.mat', {'seg_map': seg_map_img_array}) # Save the segmentation map Numpy array to .mat file.
    print('Successful') # Send the message to Matlab

try:
    run_segmentation(sys.argv[1])
except:
	print(__doc__)
	