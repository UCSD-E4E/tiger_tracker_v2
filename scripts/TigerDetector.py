#!/usr/bin/env python
import time
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn as sk
import sklearn.neighbors 
import sklearn.ensemble
import sklearn.svm
import numpy as np
import Image
import sys
import cv2
import glob

class TigerDetector():
    LABEL_PATH = None
    SAMPLE_PATH = None
    rf = None

    def __init__(self):
        self.LABEL_PATH = "../images/texture_library/texture_labels_15x15.npy"
        self.SAMPLE_PATH = "../images/texture_library/texture_samples_15x15.npy"

        train_labels = np.load(self.LABEL_PATH)
        train_imgs = np.load(self.SAMPLE_PATH)

        self.rf = sk.ensemble.RandomForestClassifier()
        self.rf.fit(train_imgs, train_labels)


    def isTiger(self, img_patch, rand_forest):
        img_patch = img_patch.flatten()
        threshold = 0
        if rand_forest.predict(img_patch) == 1:
            threshold = 255
        return threshold

    def textureMatch(self, img_in):
        img_out = np.zeros((img_in.shape[0], img_in.shape[1]))
        img_temp = np.pad(img_in, (7,7), 'constant')
        img_in_padded = img_temp[:,:, 7:10]

        for y in range(0, int(img_in.shape[0]) - 15):
            for x in range(0, int(img_in.shape[1]) - 15):
                # patch = img_in_padded[y:y+15, x:x+15]
                patch = img_in[y:y+15, x:x+15]
                #patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                #patch_hs = patch[:,:, 0:2]
                img_out[y+7, x+7] = self.isTiger(patch, self.rf)

        img_out = cv2.medianBlur(img_out.astype(np.uint8), 3)
        print np.amax(img_out)

        return img_out

