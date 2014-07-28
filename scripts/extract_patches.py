############
# Extract Positive and Negative Patches for a NN Texture library
############

import cv2
import glob
import numpy as np


PATCH_SIZE = 17

def getPatch(img):
    w = np.shape(img)[1]
    h = np.shape(img)[0]
    
    #Slice out a 16x16 texture patch at random
    x = int(np.random.rand(1,1) * (w - PATCH_SIZE))
    y = int(np.random.rand(1,1) * (h - PATCH_SIZE))
    patch = sample[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

    return patch

img_ct = 0
PATH = "/home/riley/TigerTraining/zoo_8_14_2013/webcam_video/candidate_img/"
for image in glob.glob(PATH + 'pos_814_000*.bmp'):
    print img_ct
    sample = cv2.imread(str(image))

    for i in range(0,2):
        patch = getPatch(sample)
        cv2.imwrite("./texture_library/pos_814_" + str(img_ct).zfill(6) + ".bmp", patch)
        img_ct = img_ct + 1

