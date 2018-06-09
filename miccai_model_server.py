# IMPORTS
import tensorflow as tf
import numpy as np
import os
import nibabel as nib
from scipy.io import loadmat, savemat
import random as r

import unet2 as unet
#import unet
import image_processing as impro

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nii_path = "MICCAI_data/mri/"
mat_path = "MICCAI_data/re_mri/"
label_path = "MICCAI_data/re_label/"

nii_test_path = "MICCAI_test/mri/"
mat_test_path = "MICCAI_test/re_mri/"
label_test_path = "MICCAI_test/re_label/"


#%%
train_dict = loadmat("../miccai_data/miccai_train.mat")
train_data = train_dict["data"]
train_labels = train_dict["labels"]

test_dict = loadmat("../miccai_data/miccai_test.mat")
test_data = test_dict["data"]
test_labels = test_dict["labels"]

#%%
(ims, h, w, d) = train_data.shape
train_data_tensor = np.moveaxis(train_data, -1, 0)
train_data_tensor = np.moveaxis(train_data_tensor, 0, 1)
train_data_tensor = train_data_tensor.reshape((-1, h, w))
train_data_tensor = np.expand_dims(train_data_tensor, axis = 3)

train_labels_tensor = np.moveaxis(train_labels, -1, 0)
train_labels_tensor = np.moveaxis(train_labels_tensor, 0, 1)
train_labels_tensor = train_labels_tensor.reshape((-1, h, w))

(train_data_tensor, train_labels_tensor) = impro.strip_zero_images(
        train_data_tensor, train_labels_tensor)


num_classes = 135
conv_ksize = 3
num_channels_base = 64
scale_depth = 5
learning_rate = 1e-5
bn_decay = 0.9
model_dir_name = "MICCAI_model_smsv_3"

batch_size = 16
training_steps = 5000
log_interval = 500

rounds = 6
classifier = unet.create_unet_model(train_data_tensor, num_classes, conv_ksize,
                                    num_channels_base, scale_depth,
                                    learning_rate, bn_decay, model_dir_name)

model_perf = np.zeros((rounds, 4))

for j in range(rounds):
    
    # Train classifier
    print("***** TRAINING MODEL " + str(j) + " *****")
    classifier = unet.train_unet_model(train_data_tensor, train_labels_tensor, classifier, batch_size, 
                                       training_steps, log_interval)

    print("***** EVALUATING MODEL " + str(j) + " *****")
    train_im_test = np.moveaxis(train_data[0,:,:,40:70], -1, 0)
    train_im_test = np.expand_dims(train_im_test, axis = 3)
    train_lab_test = np.moveaxis(train_labels[0,:,:,40:70], -1, 0)

    train_pred = unet.predict_unet_model(train_im_test, classifier)
    model_perf[j,0] = sum(unet.accuracy(train_pred, train_lab_test)) / train_pred.shape[0]
    model_perf[j,1] = unet.mean_dice(train_pred, train_lab_test, 135)


    test_im_test = np.moveaxis(test_data[0,:,:,40:70], -1, 0)
    test_im_test = np.expand_dims(test_im_test, axis = 3)
    test_lab_test = np.moveaxis(test_labels[0,:,:,40:70], -1, 0)
    test_pred = unet.predict_unet_model(test_im_test, classifier)
    model_perf[j,2] = sum(unet.accuracy(test_pred, test_lab_test)) / train_pred.shape[0]
    model_perf[j,3] = unet.mean_dice(test_pred, test_lab_test, 135)
    

model_perf_dict = {}
model_perf_dict["data"] = model_perf
savemat("model_perf_cb64b16it400002", model_perf_dict)




