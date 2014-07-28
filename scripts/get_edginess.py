############
# Analyze the edginess per area score for a set of images
############

import cv2
import glob
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

cv2.namedWindow("test_imgs")

edginess_scores_pos = []
edginess_scores_neg = []

for image in glob.glob('./d*.bmp'):
    #get frames as grayscale
    sample = cv2.imread(str(image), 0)
    edges = cv2.Canny(sample, 255, 255)
    edge_score = float(np.count_nonzero(edges))/(edges.shape[0]*edges.shape[1])
    edginess_scores_pos.append(edge_score)

    cv2.imshow("test_imgs", edges)
    cv2.waitKey(1)

pos_mean = np.mean(edginess_scores_pos)
pos_std = np.std(edginess_scores_pos)

print "Positive Edginess Mean: " + str(pos_mean)
print "Std: " + str(np.std(edginess_scores_pos))

for image in glob.glob('./neg*.bmp'):
    #get frames as grayscale
    sample = cv2.imread(str(image), 0)
    edges = cv2.Canny(sample, 255, 255)
    edge_score = float(np.count_nonzero(edges))/(edges.shape[0]*edges.shape[1])
    edginess_scores_neg.append(edge_score)

    cv2.imshow("test_imgs", edges)
    cv2.waitKey(1)

neg_mean = np.mean(edginess_scores_neg)
neg_std = np.std(edginess_scores_neg)

print "Negative Edginess Mean: " + str(neg_mean)
print "Std: " + str(neg_std)

x = np.linspace(-0.25, 0.25, 1000)

plt.plot(x, mlab.normpdf(x, pos_mean, pos_std))
plt.plot(x, mlab.normpdf(x, neg_mean, neg_std))

plt.show()

# Test on images:
threshold = neg_mean
false_negs = 0
false_pos = 0

for score in edginess_scores_pos:
    if edge_score < threshold:
        false_negs = false_negs + 1

for score in edginess_scores_neg:
    if edge_score > threshold:
        false_pos = false_pos + 1

print "false positives: " + str(false_pos)
print "false negatives: " + str(false_negs)
print "total samples: " + str(len(edginess_scores_neg) +
len(edginess_scores_pos))




