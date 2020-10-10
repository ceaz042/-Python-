import cv2
import numpy as np
import os

max_value = 255
max_value_H = 360
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Image'
window_detection_name = 'Frame_Threshold'
bar_window_name = 'HSV_Setting'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
save_img_name = 'turn to 1:save img'
drawContours_name = 'drawContours'
imgpath = "C:\\VScode\\Opencv\\PyCV2IP\\imgs\\banana.png"
dirname, filename = os.path.split(imgpath)
IMAGE_NAME = imgpath[:imgpath.index(".")]
OUTPUT_IMAGE = IMAGE_NAME + "_alpha.png"
print(OUTPUT_IMAGE)

global frame_show
frame  = cv2.imread(imgpath)
frame_show = cv2.imread(imgpath)

def save(val):
    if val==1:
        print('saved')
        cv2.imwrite(OUTPUT_IMAGE, frame_show)
        # cv2.waitKey(500)
    cv2.setTrackbarPos(save_img_name, bar_window_name, val)

def click_event(event, x, y, flags, param):
    global frame_show
    if event == cv2.EVENT_LBUTTONDOWN:
        contours, _ = cv2.findContours(frame_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area = []  
        # 找到最大的輪廓
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))
        # 填充最大的輪廓
        frame_show.fill(0)
        mask = cv2.drawContours(frame_show, contours, max_idx, (255, 255, 255), cv2.FILLED)
    if event == cv2.EVENT_RBUTTONDBLCLK:
        frame_show = cv2.imread(imgpath)
        print("cleaned")
        cv2.imshow(window_capture_name, frame_show)

# def drawContours(val):
#     contours, _ = cv2.findContours(frame_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     area = []
#     if val == 1:
#         # 找到最大的輪廓
#         for k in range(len(contours)):
#             area.append(cv2.contourArea(contours[k]))
#         max_idx = np.argmax(np.array(area))
#         # 填充最大的輪廓
#         mask = cv2.drawContours(frame_show, contours, max_idx, 0, cv2.FILLED)
#     if val == 0:
#         frame_show = cv2.imread(imgpath)
#     cv2.setTrackbarPos(drawContours_name, bar_window_name, val)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, bar_window_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, bar_window_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, bar_window_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, bar_window_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, bar_window_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, bar_window_name, high_V)

cv2.namedWindow(bar_window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)
cv2.createTrackbar(low_H_name, bar_window_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, bar_window_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, bar_window_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, bar_window_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, bar_window_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, bar_window_name , high_V, max_value, on_high_V_thresh_trackbar)
cv2.createTrackbar(save_img_name, bar_window_name , 0, 1, save)

while True:
    cv2.setMouseCallback(window_capture_name, click_event)
    cv2.imshow(window_capture_name, frame_show)
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    cv2.imshow(window_detection_name, frame_threshold)

    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()    