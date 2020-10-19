import cv2
import cv2IP
from matplotlib import pyplot as plt

srcImg = "C:\\VSCode\\Python\\OpenCV-Python--main\\PyCV2IP\\imgs\\front.png"
BackGround = "C:\\VSCode\\Python\\OpenCV-Python--main\\PyCV2IP\\imgs\\ocean.png"

aa = cv2IP.AlhpaBlend()

Alpha = aa.SplitAlpha(srcImg)
aa.DoBlending(srcImg, BackGround, Alpha)

# img = cv2IP.BaseIP.ImRead(srcImg)
# mask = cv2IP.HistIP.ImBGR2Gray(img)
# GrayHist = cv2IP.HistIP.CalcGrayHist(mask)
# plt.plot(GrayHist)
# plt.show()