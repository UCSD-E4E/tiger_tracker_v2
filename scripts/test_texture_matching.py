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

import TigerDetector

td = TigerDetector.TigerDetector()

cv2.namedWindow("image")
cv2.namedWindow("thresh")

#img = cv2.imread("./tiger_nsf2.bmp")
#img_thresh = td.textureMatch(img)
#cv2.imshow("image", img)
#cv2.imshow("thresh", img_thresh)

cv2.waitKey(0)

for image in glob.glob("../images/*.bmp"):
    img = cv2.imread(str(image))
    img_thresh = td.textureMatch(img)
    cv2.imshow("image", img)
    cv2.imshow("thresh", img_thresh)

    cv2.waitKey(0)


