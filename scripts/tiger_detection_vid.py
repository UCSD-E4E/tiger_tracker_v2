import cv2
import numpy as np
import TigerDetector as td


cv2.namedWindow("Raw")
cv2.namedWindow("Tiger Mask")


detector = td.TigerDetector()

video_in = cv2.VideoCapture("./demo_vid.avi")

got_f, frame = video_in.read()

while got_f == True:
    img_thresh = detector.textureMatch(np.array(frame))
    cv2.imshow("Tiger Mask", img_thresh)
    cv2.imshow("Raw", frame)
    cv2.waitKey(1)
    got_f, frame = video_in.read()
