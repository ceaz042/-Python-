import cv2
import numpy as np
import enum
import os

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

class AlhpaBlend:
    def __init__(self):
        self.sign = 0
        self.lowH = 0
        self.lowS = 0
        self.lowV = 0
        self.high_H = 360
        self.high_S = 255
        self.high_V = 255
    def SplitAlpha(self, SrcImg):
        window_capture_name = 'Image'
        window_detection_name = 'Frame_Threshold'
        bar_window_name = 'HSV_Setting'
        low_H_name = 'Low H'
        low_S_name = 'Low S'
        low_V_name = 'Low V'
        high_H_name = 'High H'
        high_S_name = 'High S'
        high_V_name = 'High V'
        save_img_name = 'make Alpha'
        frame  = cv2.imread(SrcImg)
        self.frame_show = cv2.imread(SrcImg)
        def save(val):
            if val==1:
                self.sign +=1
            cv2.setTrackbarPos(save_img_name, bar_window_name, val)
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # self.contours_flag = 1
                # print(self.contours_flag)
                contours, _ = cv2.findContours(frame_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                area = []  
                contours_num = len(contours)
                # 找到最大的輪廓
                for k in range(contours_num):
                    area.append(cv2.contourArea(contours[k]))
                max_idx = np.argmax(np.array(area))
                # 填充最大的輪廓
                self.frame_show.fill(0)
                mask = cv2.drawContours(self.frame_show, contours, max_idx, (255, 255, 255), cv2.FILLED)
            if event == cv2.EVENT_RBUTTONDBLCLK:
                self.frame_show = cv2.imread(SrcImg)
                print("cleaned")
                cv2.imshow(window_capture_name, self.frame_show)
        def on_low_H_thresh_trackbar(val):
           self.lowH = val
           self.lowH = min(self.high_H-1, self.lowH)
           cv2.setTrackbarPos(low_H_name, bar_window_name, self.lowH)
        def on_high_H_thresh_trackbar(val):
            self.high_H = val
            self.high_H = max(self.high_H, self.lowH+1)
            cv2.setTrackbarPos(high_H_name, bar_window_name, self.high_H)
        def on_low_S_thresh_trackbar(val):
            self.lowS = val
            self.lowS = min(self.high_S-1, self.lowS)
            cv2.setTrackbarPos(low_S_name, bar_window_name, self.lowS)
        def on_high_S_thresh_trackbar(val):
            self.high_S = val
            self.high_S = max(self.high_S, self.lowS+1)
            cv2.setTrackbarPos(high_S_name, bar_window_name, self.high_S)
        def on_low_V_thresh_trackbar(val):
            self.lowV = val
            self.lowV = min(self.high_V-1, self.lowV)
            cv2.setTrackbarPos(low_V_name, bar_window_name, self.lowV)
        def on_high_V_thresh_trackbar(val):
            self.high_V = val
            self.high_V = max(self.high_V, self.lowV+1)
            cv2.setTrackbarPos(high_V_name, bar_window_name, self.high_V)
        cv2.namedWindow(bar_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_capture_name)
        cv2.namedWindow(window_detection_name)
        cv2.createTrackbar(low_H_name, bar_window_name , self.lowH, 360, on_low_H_thresh_trackbar)
        cv2.createTrackbar(high_H_name, bar_window_name , self.high_H, 360, on_high_H_thresh_trackbar)
        cv2.createTrackbar(low_S_name, bar_window_name , self.lowS, 255, on_low_S_thresh_trackbar)
        cv2.createTrackbar(high_S_name, bar_window_name , self.high_S, 255, on_high_S_thresh_trackbar)
        cv2.createTrackbar(low_V_name, bar_window_name , self.lowV, 255, on_low_V_thresh_trackbar)
        cv2.createTrackbar(high_V_name, bar_window_name , self.high_V, 255, on_high_V_thresh_trackbar)
        cv2.createTrackbar(save_img_name, bar_window_name , 0, 1, save)
        while True:
            cv2.setMouseCallback(window_capture_name, click_event)
            cv2.imshow(window_capture_name, self.frame_show)
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(frame_HSV, (self.lowH, self.lowS, self.lowV), (self.high_H, self.high_S, self.high_V))

            cv2.imshow(window_detection_name, frame_threshold)
            if self.sign==1:
                Alpha = self.frame_show.copy()
                break

            key = cv2.waitKey(30)
            if key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()
        return Alpha
    def DoBlending(self, Foreground, Background, Alpha):
        Foreground = cv2.imread(Foreground)
        Background = cv2.imread(Background)
        # Convert uint8 to float
        foreground = Foreground.astype(float)
        background = Background.astype(float)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = Alpha.astype(float)/255
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)
        # Display image
        cv2.imshow("outImg", outImage/255)
        cv2.waitKey(0)

class HistIP(BaseIP):
    def __init__(self):

    def ImBGR2Gray(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)

    def ImBGRA2BGR(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGRA2BGR)

    def CalcGrayHist(self, SrcGray):
        return cv2.calcHist([SrcGray], [0], None, [256], [0, 256])

    def ShowGrayHist(self, winname, GrayHist):

    def CalcColorHist(self, SrcColor):
        color = ('b','g','r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0, 256])
        return histr

    def ShowColorHist(self, winname, ColorHist):

       