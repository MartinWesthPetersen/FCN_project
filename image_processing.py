# Imports
import numpy as np
import tensorflow as tf
import cv2 as cv
import math as m
##import matplotlib.pyplot as plt
import os
import nibabel as nib
from scipy.io import loadmat, savemat
    
def nii_to_mat(nii_path, mat_path):
    nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
    files = os.listdir(nii_path)
    if not os.path.exists(mat_path):
        os.makedirs(mat_path)
    for file in files:
        print("Processing file: " + file)
        name, ext = os.path.splitext(os.path.split(file)[-1])
        data = nib.load(nii_path + file).get_data().squeeze()
        print(data.shape)
        data_dict = {}
        data_dict["data"] = data
        savemat(mat_path + name + ".mat", data_dict)

def prep_data(data_path, label_path):
    def normalize_intensity_slicewise(data):
        (b, h, w, d) = data.shape
        data_norm = np.zeros_like(data).astype(np.float32)
        for i in range(b):
            for j in range(d):
                I = data[i,:,:,j]
                if (len(np.unique(I)) == 1):
                    continue
                mean = np.mean(I)
                std = np.std(I)
                data_norm[i,:,:,j] = np.divide(np.subtract(I, mean), std)
        return data_norm

    def normalize_intensity(data):
        (b, h, w, d) = data.shape
        data_norm = np.zeros_like(data).astype(np.float32)
        for i in range(b):
            I = data[i,:,:,:]
            mean = np.mean(I)
            std = np.std(I)
            data_norm[i,:,:,:] = np.divide(np.subtract(I, mean), std)
        return data_norm
    
    data_files = os.listdir(data_path)
    max_depth = 0
    
    print("*** Aligning data dimensions ***")
    for file in data_files:
        (h, w, mat_depth) = loadmat(data_path + file)["data"].shape
        max_depth = mat_depth if mat_depth > max_depth else max_depth
    
    data = np.zeros((len(data_files), h, w, max_depth), dtype = np.uint16)
    for i in range(0, len(data_files)):
        file = data_files[i]
        mat = loadmat(data_path + file)["data"]
        data[i, :, :, :mat.shape[2]] = mat
    
    print("*** Aligning label dimensions ***")
    label_files = os.listdir(label_path)
    labels = np.zeros((len(label_files), h, w, max_depth), dtype = np.uint8)
    for i in range(0, len(label_files)):
        file = label_files[i]
        mat = loadmat(label_path + file)["label"]
        labels[i, :, :, :mat.shape[2]] = mat
    
    print("*** Normalize data intensities ***")
    data_norm = normalize_intensity(data)
    return (data_norm, labels)


def strip_zero_images(data, labels):
    (b,h,w,c) = data.shape
    num_z = 0
    for i in range(0, b):
        if (len(np.unique(labels[i,:,:])) <= 1):
            num_z += 1
    new_data = np.zeros((b - num_z, h, w, c))
    new_labels = np.ones((b - num_z, h, w))
    j = 0
    for i in range(0, b):
        if (len(np.unique(labels[i,:,:])) >= 2):
            new_data[j,:,:,:] = data[i,:,:,:]
            new_labels[j,:,:] = labels[i,:,:]
            j += 1
    return (new_data, new_labels)



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
    
    
