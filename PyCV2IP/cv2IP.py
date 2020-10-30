import cv2
import numpy as np
import enum
import os
from matplotlib import pyplot as plt

class BaseIP:
    @staticmethod
    def ImRead(SrcImg):
          return cv2.imread(SrcImg, cv2.IMREAD_UNCHANGED)
        
    @staticmethod
    def ImWrite(SrcImg, img):
        cv2.imwrite(SrcImg, img[cv2.IMWRITE_JPEG_QUALITY, 100])

    @staticmethod
    def ImShow(winName, img):
        cv2.imshow(winName, img)

    @staticmethod
    def ImWindow(winName):
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

class AlhpaBlend(BaseIP):
    @staticmethod
    def SplitAlpha(SrcImg):
        channel = []
        channel = cv2.split(SrcImg)
        Foreground = cv2.merge((channel[0], channel[1], channel[2]))
        Alpha = cv2.merge((channel[3], channel[3], channel[3]))
        return Foreground, Alpha
  
    @staticmethod
    def DoBlending(Foreground , Background, Alpha, Visible):
        # Convert uint8 to float 
        foreground = Foreground.astype(float)
        background = Background.astype(float)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = Alpha.astype(float)/255
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - (alpha*Visible), background)
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)
        return outImage/255

# class HistIP(BaseIP):
#     @staticmethod
#     def ImBGR2Gray(SrcImg):
#         return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)

#     @staticmethod
#     def ImBGRA2BGR(SrcImg):
#         return cv2.cvtColor(SrcImg, cv2.COLOR_BGRA2BGR)

#     @staticmethod
#     def CalcGrayHist(SrcGray):
#         return cv2.calcHist(SrcGray, [0], None, [256], [0, 256])
#     @staticmethod
#     def ShowGrayHist(Winname, GrayHist):
#         BaseIP.ImWindow(Winname)
#         BaseIP.ImShow(Winname, GrayHist)
    
class HistIP(BaseIP):
    def __init__(self):
        self.__H = 260
        self.__W = 512

    def ImBGR2Gray(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)

    def ImBGRA2BGR(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGRA2BGR)

    def CalcGrayHist(self, SrcGray):
        return cv2.calcHist(SrcGray, [0], None, [256], [0, 256])

    def ShowGrayHist(self, Winname, GrayHist):
        Hist = np.zeros((self.__H, self.__W, 3), np.uint8)
        pixel = 0
        for i in range(len(GrayHist)):
            cv2.line(Hist, (pixel, self.__H), (pixel, self.__H-GrayHist[i]), (125, 125, 125), 2)
            pixel += 2
            if pixel > self.__W:
                pixel = self.__W
        BaseIP.ImShow(Winname, Hist)

    def CalcColorHist(self, SrcColor):
        # #宣告空的NP陣列
        # ColorHist = np.empty((0, 0))
        Color_B = cv2.calcHist(SrcColor,[0],None,[256],[0, 256])
        Color_G = cv2.calcHist(SrcColor,[1],None,[256],[0, 256])
        Color_R = cv2.calcHist(SrcColor,[2],None,[256],[0, 256])
        Color = np.r_[Color_B, Color_G]
        ColorHist = np.r_[Color, Color_R]
        #尋找NUMPY的加法
        return ColorHist

    def ShowColorHist(self, Winname, ColorHist):
        Hist = np.zeros((self.__H, self.__W, 3), np.uint8)
        pixel = 0
        for i in range(len(ColorHist)):
            cv2.line(Hist, (pixel, self.__H), (pixel, self.__H-ColorHist[i]), (125, 0, 0), 2)
            pixel += 2
            if pixel > self.__W:
                pixel = self.__W
        BaseIP.ImShow(Winname, Hist)
