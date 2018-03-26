# IMPORTS
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import cv2 as cv

import unet
import image_processing as impro

# Determines the device to run on
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = "FASSEG-repository-master/V2/Test_RGB/"
label_path = "FASSEG-repository-masterV2/Test_Labels/"

test_data_path = "FASSEG-repository-masterV2/Train_RGB/"
test_label_path = "FASSEG-repository-masterV2/Train_Labels/"

channels = 3
#%% Reformat label images from colors to labels, storing in numpy files.

#label_codes = [(255,0,0), (127,0,0), (255,255,0), (0,0,255), (0,255,255), (0,255,0)]
#(data, relabels) = impro.relabel(data_path, label_path, '.bmp', '.bmp', channels, 
#                         label_codes, fix_h = 512, fix_w = 352)
#np.save("FASSEG_data", data)
#np.save("FASSEG_labels", relabels)

#label_codes = [(255,0,0), (127,0,0), (255,255,0), (0,0,255), (0,255,255), (0,255,0)]
#(test_data, test_labels) = impro.relabel(test_data_path, test_label_path,
#'.bmp', '.bmp', channels, label_codes, fix_h = 512, fix_w = 352)
#np.save("FASSEG_test_data", test_data)
#np.save("FASSEG_test_labels", test_labels)

#%% Loading premade data and labels for training and test
train_data = np.load('FASSEG_data.npy')
train_labels = np.load('FASSEG_labels.npy')

test_data = np.load('FASSEG_test_data.npy')
test_labels = np.load('FASSEG_test_labels.npy')

# Model parameters
num_classes = 6
conv_ksize = 5
num_channels_base = 16
scale_depth = 5
learning_rate = 1e-5
model_dir_name = "FASSEG_model1_3"

classifier = unet.create_unet_model(train_data, num_classes, conv_ksize, 
                                    num_channels_base, scale_depth, 
                                    learning_rate, model_dir_name)


#%% Perform training on GPU
#batch_size = 1
#training_steps = 2
#log_interval = 1
#
#classifier = unet.train_unet_model(train_data, train_labels, classifier, batch_size, 
#                                   training_steps, log_interval)

#%% Evaluate model
# Training scores
train_predictions = unet.predict_unet_model(train_data, classifier)
train_scores = unet.eval_pred(train_predictions, train_labels, 6)
train_avg_score = (1 / len(train_scores)) * np.sum(train_scores)
print(train_scores)
print(train_avg_score)

#%%
# Test scores
test_predictions = unet.predict_unet_model(test_data, classifier)
test_scores = unet.eval_pred(test_predictions, test_labels, 6)
avg_score = (1 / len(test_scores)) * np.sum(test_scores)
print(test_scores)
print(avg_score)
