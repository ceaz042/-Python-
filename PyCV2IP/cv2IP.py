import cv2
import numpy
import enum

class BaseIP:
    def __init__(self, img):
        self.img = img

    @staticmethod
    def ImRead(filename):
          return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        
    @staticmethod
    def ImWrite(filename, img):
        cv2.imwrite(filename, img[cv2.IMWRITE_JPEG_QUALITY, 100])

    @staticmethod
    def ImShow(winName, img):
        cv2.imshow(winName, img)

    @staticmethod
    def ImWindow(winName):
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
