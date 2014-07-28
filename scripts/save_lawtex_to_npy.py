import scipy
import numpy as np
import glob
import cv2
import TextureEnergy as TexEn

IMAGE_PATH = "./"
IMG_SIZE = 16*16

neg_total = 0
pos_total = 0

for sample in glob.glob('./pos*.bmp'):
    pos_total = pos_total + 1


for sample in glob.glob('./neg*.bmp'):
    neg_total = neg_total + 1


texture_vectors = np.zeros((pos_total + neg_total, 9))
texture_labels = np.zeros((neg_total + pos_total, 1))

te = TexEn.TextureEnergy()
te.test_law_tex()

i = 0
order = np.random.permutation(pos_total + neg_total)
for sample in glob.glob('./pos*.bmp'):
    img = cv2.imread(str(sample), 0)
    texture_vectors[order[i], :] = te.getTexture(img)#img.astype(np.float32))
    texture_labels[order[i]] = 1
    i = i + 1


for sample in glob.glob('./neg*.bmp'):
    img = cv2.imread(str(sample), 0)
    texture_vectors[order[i], :] = te.getTexture(img)#img.astype(np.float32))
    texture_labels[order[i]] = 0
    i = i + 1

total = np.zeros((1,9))
for i in range(0, len(texture_labels)):
    if texture_labels[i] == 1:
        total = total + texture_vectors[i,:]

print "Tiger mean = " + str(total/pos_total)

total = np.zeros((1,9))
for i in range(0, len(texture_labels)):
    if texture_labels[i] == 0:
        total = total + texture_vectors[i,:]

print "non-Tiger mean = " + str(total/neg_total)

np.save("law_texture_labels.npy", texture_labels)
np.save("law_texture_samples.npy", texture_vectors)

