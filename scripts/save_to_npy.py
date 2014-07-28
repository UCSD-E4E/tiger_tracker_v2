# save_to_npy.py
# Saves a directory full of sample image patches into a .npy binary for quick
# loading and access in python

import scipy
import numpy as np
import glob
import cv2

IMAGE_PATH = "./"
IMG_SIZE = 15*15

neg_total = 0
pos_total = 0

for sample in glob.glob('../images/pos*.bmp'):
    pos_total = pos_total + 1


for sample in glob.glob('../images/neg*.bmp'):
    neg_total = neg_total + 1


texture_vectors = np.zeros((pos_total + neg_total, IMG_SIZE * 2))
texture_labels = np.zeros((neg_total + pos_total, 1))

i = 0
order = np.random.permutation(pos_total + neg_total)
for sample in glob.glob('../images/pos*.bmp'):
    img = cv2.imread(str(sample))
 #   img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  #  img_hs = img[:,:, 0:2]
    texture_vectors[order[i], :] = img_hs[0:15, 0:15].flatten()
    texture_labels[order[i]] = 1
    i = i + 1


for sample in glob.glob('../images/neg*.bmp'):
    img = cv2.imread(str(sample))
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    img_hs = img[:,:,0:1]
#    texture_vectors[order[i], :] = img_hs[0:15, 0:15].flatten()
    #img = cv2.imread(str(sample))
    texture_vectors[order[i], :] = img[0:15, 0:15].flatten()
    texture_labels[order[i]] = 0
    i = i + 1


np.save("../images/texture_labels_15x15.npy", texture_labels)
np.save("../images/texture_samples_15x15.npy", texture_vectors)

