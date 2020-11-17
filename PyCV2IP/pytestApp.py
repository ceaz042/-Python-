import numpy as np
import cv2
import cv2IP

def Mid_Project():
    ip = cv2IP.HistIP()
    srcImg_dir = "C:\\VSCode\\OpenCV\\PyCV2IP\\imgs\\aspens_in_fall.jpg"
    refImg_dir = "C:\\VSCode\\OpenCV\\PyCV2IP\\imgs\\forest-resized.jpg"
    srcImg = ip.ImRead(srcImg_dir)
    refImg = ip.ImRead(refImg_dir)

    outImg = ip.HistMatching(srcImg, refImg, cv2IP.ColorType.USE_YUV)
    
    ip.ImShow("ref img", refImg)
    ip.ImShow("src img", srcImg)
    ip.ImShow("out img", outImg)
    O_Hist = ip.CalcColorHist(srcImg)
    ip.ShowColorHist("Original Color Hist", O_Hist)
    out_Hist = ip.CalcColorHist(outImg)
    ip.ShowColorHist("Hist after matching", out_Hist)
    del ip

if __name__ == '__main__':
    Mid_Project()
    cv2.waitKey(0)
