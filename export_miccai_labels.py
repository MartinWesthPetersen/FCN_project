"""
This script laods the MICCAI 2012 Multi-Atlas Challenge dataset stored in nifty format (.nii)
and exports the labels to mat files (.mat).

Note that labels are converted into 135 categories.
"""
#%%
import os
import glob
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from scipy.io import loadmat, savemat
#%%
# See Miccai rules
nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
ignored_labels = list(range(1,4))+list(range(5,11))+list(range(12,23))+list(range(24,30))+[33,34]+[42,43]+[53,54]+list(range(63,69))+[70,74]+\
                    list(range(80,100))+[110,111]+[126,127]+[130,131]+[158,159]+[188,189]

# 47: right hippocampus, 48: left hippocampus, 0: others
true_labels = [4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57,
                58, 59, 60, 61, 62, 69, 71, 72, 73, 75, 76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129, 132, 133, 134, 135, 136,
                137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
                179, 180, 181, 182, 183, 184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                201, 202, 203, 204, 205, 206, 207]


def label_filtering(lab, ignore_labels, true_labels):
    for ignored_label in ignored_labels:
        lab[lab == ignored_label] = 0
    for idx, label in enumerate(true_labels):
        lab[lab==label] = idx+1
    return lab

##
# label_root_dir = './datasets/miccai/test/label/'
# out_dir = './datasets/miccai/test/label_mat/'
#label_root_dir = './augmented/aug1/label/'
#out_dir = './augmented/aug1/refactored_label/'
label_root_dir = "MICCAI_test/label/"
out_dir = "MICCAI_test/re_label/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

label_files = glob.glob(label_root_dir+'*.nii')
if not label_files:
    label_files = glob.glob(label_root_dir+'*.mat')

for label_file in label_files:
    print('Processing {}'.format(label_file))
    label_name, label_ext = os.path.splitext(os.path.split(label_file)[-1])
    out_file = os.path.join(out_dir, label_name) + '.mat'

    if label_ext == ".nii":
        lab = nib.load(label_file).get_data().squeeze()
    else:
        lab = loadmat(label_file)
        name = [n for n in lab if isinstance(lab[n], np.ndarray)]
        assert len(name) == 1
        lab = lab[name[0]].squeeze()
    lab = lab.astype(int, copy=False)
    lab = label_filtering(lab, ignored_labels, true_labels)
    print(lab.shape)

    lab_dict = {}
    lab_dict['label'] = lab
    savemat(out_file, lab_dict)
