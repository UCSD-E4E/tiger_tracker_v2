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

IMG_SIZE = 28*28

LABEL_PATH = "../images/texture_library/texture_labels_15x15.npy"
SAMPLE_PATH = "../images/texture_library/texture_samples_15x15.npy"

train_labels = np.load(LABEL_PATH)
img_train = np.load(SAMPLE_PATH)

set_size = np.size(train_labels)
print "Total Set Size: " + str(set_size)
print np.shape(img_train)

train_size = (int) (set_size * 0.5)
test_size = set_size - train_size

train_range = range(0, train_size)
test_range = range(train_size, set_size)

def showTrainImage(test_image):
    img = test_image.reshape(28,28)
    plt.imshow(img, cmap = cm.Greys_r)
    plt.show()


#benchmark K Nearest Neighbors
def benchmarkKNN(train_imgs, train_labels, test_imgs, test_labels):
    knn = sk.neighbors.KNeighborsClassifier(n_neighbors = 5, weights='uniform', 
                                            algorithm='auto', leaf_size=30, 
                                            warn_on_equidistant=True, p=2)

    start_t = time.time()
    knn.fit(train_imgs, train_labels)
    train_t = time.time() - start_t;
    print "KNN train time: " + str(train_t)
    print "KNN training complete, beginning test set:"

    print np.shape(train_labels)
    error_count = 0
    start_t = time.time()
    for  i in range(0, len(test_labels)):
        if knn.predict(test_imgs[i, :]) != test_labels[i]:
            error_count = error_count + 1
        if i % 100 == 0:
            sys.stdout.write("\r")
            sys.stdout.write(str((100.0*(i)/len(test_labels))) + 
                              "% complete (KNN)")

    error_rate = 100.0*error_count/len(test_labels)
    test_t = time.time() - start_t
    print "Test time = " + str(test_t)

    print "KNN Performance [Error %, train_t, test_t]: " 
    print error_rate 
    print train_t
    print test_t
    return error_rate, train_t, test_t

# Benchmark Random Forest Classifier
def benchmarkRF(train_imgs, train_labels, test_imgs, test_labels):
    rf = sk.ensemble.RandomForestClassifier()

    start_t = time.time()
    rf.fit(train_imgs, train_labels)
    train_t = time.time() - start_t

    error_count = 0
    print "Test Size = " + str(len(test_labels))

    start_t = time.time()
    for i in range(0, len(test_labels)):
        if rf.predict(test_imgs[i, :]) != test_labels[i]:
            error_count = error_count + 1
        if (i % 100) == 0:
            sys.stdout.write("\r")
            sys.stdout.write(str((100.0*(i)/len(test_labels))) +
                              "% complete (RF)")

    test_t = time.time() - start_t
    error_rate = 100.0 * error_count/(len(test_labels))

    print "RF Performance [Error %, train_t, test_t]: "
    print error_rate
    print train_t
    print test_t
    return error_rate, train_t, test_t


# Benchmark SVM
def benchmarkSVM(train_imgs, train_labels, test_imgs, test_labels):
    clf = sk.svm.SVC()

    start_t = time.time()
    clf.fit(train_imgs, train_labels)
    train_t = time.time() - start_t    

    error_count = 0
    print "Test Size = " + str(len(test_labels))
    
    start_t = time.time()
    for i in range(0, len(test_labels)):
        if clf.predict(test_imgs[i, :]) != test_labels[i]:
            error_count = error_count + 1
        if (i % 100) == 0:
            sys.stdout.write("\r")
            sys.stdout.write(str((100.0*(i)/len(test_labels))) + 
                              "% complete (CLF)")
    
    test_t = time.time() - start_t
    error_rate = 100.0 * error_count/(len(test_labels))
    
    print "SVM Performance [Error %, train_t, test_t]: " 
    print error_rate 
    print train_t
    print test_t
    return error_rate, train_t, test_t




err_rf, train_t_rf, test_t_rf = benchmarkRF(img_train[train_range, :], 
                                            train_labels[train_range], 
                                            img_train[test_range, :], 
                                            train_labels[test_range])

err_knn, train_t_knn, test_t_knn = benchmarkKNN(img_train[train_range, :], 
                                                train_labels[train_range], 
                                                img_train[test_range, :], 
                                                train_labels[test_range])

err_svm, train_t_svm, test_t_svm = benchmarkSVM(img_train[train_range, :], 
                                            train_labels[train_range], 
                                            img_train[test_range, :], 
                                            train_labels[test_range])


