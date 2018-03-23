# Imports
import numpy as np
import tensorflow as tf
import cv2 as cv
import math as m
import matplotlib.pyplot as plt
import os
    
def crop_images_to_smallest(image_path, label_path, im_type, lab_type, channels,
                            fix_h=0, fix_w=0):
    im_names = os.listdir(image_path)
    
    if (fix_h == 0 and fix_w == 0):
        # Find smallest dimensions
        min_h = m.inf; min_w = m.inf
        for name in im_names:
            im = cv.imread(image_path + name)
            (h,w,d) = im.shape
            min_h = min(h,min_h); min_w = min(w,min_w)
    else:
        min_h = fix_h
        min_w = fix_w
    ims = np.zeros((len(im_names), min_h, min_w, channels), dtype=np.uint8)
    labs = np.zeros((len(im_names), min_h, min_w, channels), dtype=np.uint8)
    for i in range(0, len(im_names)):
        name = im_names[i]
        im = cv.imread(image_path + name.split('.')[0] + im_type)
        label = cv.imread(label_path + name.split('.')[0] + lab_type)
        (h,w,d) = im.shape
        start_h = (h - min_h) // 2
        start_w = (w - min_w) // 2
        im_crop = im[start_h : start_h + min_h, start_w : start_w + min_w, :]
        label_crop = label[start_h : start_h + min_h, start_w : start_w + min_w, :]
        im_crop = cv.cvtColor(im_crop, cv.COLOR_BGR2RGB)
        label_crop = cv.cvtColor(label_crop, cv.COLOR_BGR2RGB)
        ims[i,:,:,:] = im_crop
        labs[i,:,:,:] = label_crop
    return (ims, labs, min_h, min_w)

    
def relabel(data_path, label_path, data_type, label_type, channels, label_codes,
            fix_h = 0, fix_w = 0):
    (data, labels, height, width) = crop_images_to_smallest(data_path, 
    label_path, data_type, label_type, channels = channels, fix_h = fix_h, 
    fix_w = fix_w)
    
    label_key_values = zip(label_codes, range(0, len(label_codes)))
    label_dict = {key : value for (key, value) in label_key_values}
    
    relabels = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2]))
    
    for b in range(0, labels.shape[0]):
        print(str(b) + "/" + str(labels.shape[0]))
        for h in range(0, height):
            for w in range(0, width):
                relabels[b, h, w] = label_dict.get(tuple(labels[b, h, w, :]), 2)
    
    return (data, relabels)
    
    