import numpy as np
import scipy.signal as spsig
import cv2

class TextureEnergy():
    L5 = np.transpose([[ 1, 4, 6, 4, 1 ]]).astype(np.float32)
    E5 = np.transpose([[-1, -2, 0, 2, 1]]).astype(np.float32)
    S5 = np.transpose([[-1, 0, 2, 0, -1]]).astype(np.float32)
    R5 = np.transpose([[1, -4, 6, -4, 1]]).astype(np.float32)
    ksize = 7

    def __init__(self):
        self.init = True

    def getEnergy(self, region):
        mag = np.absolute(region)
        energy = np.sum(mag)
        return energy

    def getMask(self, F1, F2):
        mask = np.dot(F1, F2.T)
        return mask

    def getHomogenous(self, image, F1):
        mask = self.getMask(F1, F1)
        filtered = cv2.filter2D(image, -1, mask)
        return self.getAvgEnergy(filtered, self.ksize)
    
    def getHeterogenous(self, image, F1, F2):
        mask = self.getMask(F1, F2)
        filtered = cv2.filter2D(image, -1, mask)
        F1F2 = self.getAvgEnergy(filtered, self.ksize)
        mask = self.getMask(F2, F1)
        filtered = cv2.filter2D(image, -1, mask) 
        F2F1 = self.getAvgEnergy(filtered, self.ksize)
        return (F1F2 + F2F1)/2.0

    def remove_illumination(self, image,ksize):
        # subtract the local average from each pixel
        mask = np.ones((ksize, ksize))
        mask[ksize/2, ksize/2] = -1
        mask = mask / (ksize * ksize)

        img_out = cv2.filter2D(image, -1, mask)
        return img_out

    def getAvgEnergy(self, img_in, ksize):
        img_abs = np.absolute(img_in)
        energy_img = cv2.filter2D(img_in, -1, np.ones((ksize, ksize)))
        return np.mean(energy_img)

    def getTexture(self, image):
        law_tex = []
        image = self.remove_illumination(image, self.ksize)
        law_tex.append(self.getHomogenous(image, self.E5))
        law_tex.append(self.getHomogenous(image, self.S5))
        law_tex.append(self.getHomogenous(image, self.R5))
        law_tex.append(self.getHeterogenous(image, self.E5, self.L5))
        law_tex.append(self.getHeterogenous(image, self.S5, self.L5))
        law_tex.append(self.getHeterogenous(image, self.R5, self.L5))
        law_tex.append(self.getHeterogenous(image, self.S5, self.E5))
        law_tex.append(self.getHeterogenous(image, self.R5, self.E5))
        law_tex.append(self.getHeterogenous(image, self.R5, self.S5))

        # return a nice numpy array
        x = np.zeros((1,9))
        x[0,:] = law_tex
        return x

    def test_law_tex(self):
        E5L5 = np.array([[-1, -4, -6, -4, -1],
                         [-2, -8, -12, -8, -4],
                         [ 0,  0,  0,  0,  0],
                         [2, 8, 12, 8, 4],
                         [1, 4, 6, 4, 1]]).astype(np.float32)
                        
        print self.getMask(self.E5, self.L5)
        print 
        print E5L5
