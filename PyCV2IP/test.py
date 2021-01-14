import cv2
import cv2IP
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

from matplotlib import pyplot as plt

# channel = []
# Foreground = cv2.imread(srcImg,cv2.IMREAD_UNCHANGED)
# Back =  cv2.imread(BackGround,cv2.IMREAD_UNCHANGED)
# channel = cv2.split(Foreground)
# Foreground = cv2.resize(Foreground, (1280, 720),  interpolation=cv2.INTER_CUBIC)
# foreground = Foreground.astype(float)
# background = Back.astype(float)
# Alpha = cv2.merge((channel[3], channel[3], channel[3]))
# alpha = Alpha.astype(float)/255
# cv2.imshow('A', Alpha)
# alpha = cv2.resize(alpha,(1280,720))
# print(alpha.shape,foreground.shape)
# foreground = cv2.multiply(alpha, foreground[:,:,:3])
# background = cv2.multiply(1.0 - alpha, background)
# outImage = cv2.add(foreground, background)
# cv2.imshow("outImg", outImage/255)
# cv2.waitKey(0)

def Example_AlphaBlend():
    global Visible
    Visible = 0
    aa = cv2IP.AlhpaBlend()
    img = aa.ImRead(srcImg)
    back = aa.ImRead(BackGround)
    fore, alpha = aa.SplitAlpha(img)
    ImDim = np.shape(img)
    bar_name = 'Visible_Setting'
    def Visible_Setting(val):
        global Visible
        Visible = val
        cv2.setTrackbarPos(bar_name, "AlphaBlending Result", Visible)
    aa.ImWindow("AlphaBlending Result")
    cv2.createTrackbar(bar_name, "AlphaBlending Result", 0, 100, Visible_Setting)
    if (ImDim[0] !=back.shape[0] or ImDim[1] !=back.shape[1]):
        back = cv2.resize(back, (ImDim[1], ImDim[0]))
    while(1):        
        out = aa.DoBlending(fore, back, alpha, Visible/100)
        aa.ImShow("AlphaBlending Result", out)
        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()
    del aa
    return

def Example_ColorHistEqualize_Original(CType):
    Hist = cv2IP.HistIP()
    img = Hist.ImRead(srcImg)
    img = cv2.resize(img, (1280, 720))
    F_eq = Hist.ColorEqualize(img, CType)
    Title = "ForeGround Color"
    EQ_Title = "ForeGround Color Equalized"
    Hist.ImWindow(Title)
    Hist.ImShow(Title, img)
    Hist.ImWindow(EQ_Title)
    Hist.ImShow(EQ_Title, F_eq)

    F_Hist = Hist.CalcColorHist(img)
    Hist.ShowColorHist("Foreground Color Hist", F_Hist)
    Feq_Hist = Hist.CalcColorHist(F_eq)
    Hist.ShowColorHist("Foreground Color Equalized Hist", Feq_Hist)
    del Hist

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

def Example_Smooth(smType):
    ip = cv2IP.ConvIP()
    img = ip.ImRead(srcImg)
    ip.ImShow("original image", img)
    outImg = ip.Smooth2D(img, 5, smType)
    ip.ImShow("smoothed image -5", outImg)
    outImg = ip.Smooth2D(img, 15, smType)
    ip.ImShow("smoothed image -15", outImg)
    del ip

def Example_ImEdge(EdType):
    ip = cv2IP.ConvIP()
    Img = ip.ImRead(srcImg)
    outImg = ip.EdgeDetect(Img, EdType)
    if (EdType == cv2IP.EdgeType(1)):
        ip.ImShow("Sobel Edge", outImg)
    elif (EdType == cv2IP.EdgeType(2)):
        ip.ImShow("Canny Edge", outImg)
    elif (EdType == cv2IP.EdgeType(3)):
        ip.ImShow("Scharr Edge", outImg)
    elif (EdType == cv2IP.EdgeType(4)):
        ip.ImShow("Laplacian Edge", outImg)
    elif (EdType == cv2IP.EdgeType(5)):
        ip.ImShow("Color Sobel Edge", outImg)
    del ip

def Example_ImConv2D_Roberts():
    ip = cv2IP.ConvIP()
    Img = ip.ImRead(srcImg)    
    ip.ImShow("original", Img)
    src_gray = ip.ImBGR2Gray(Img)
    kernels = ip.GetRobertsKernel()
    grad_planes = []
    for i in range(0, len(kernels)):
        grad_planes.append(ip.Conv2D(src_gray, kernels[i]))
        grad_planes[i] = cv2.convertScaleAbs(grad_planes[i])

    GradImg = cv2.addWeighted(grad_planes[0], 0.5, grad_planes[i], 0.5, 0)
    ip.ImShow("Rober Images", GradImg)
    del ip

def Example_ImConv2D_Prewitt():
    ip = cv2IP.ConvIP()
    Img = ip.ImRead(srcImg)    
    ip.ImShow("original", Img)
    src_gray = ip.ImBGR2Gray(Img)
    kernels = ip.GetPrewittKernel()
    grad_planes = []
    for i in range(0, len(kernels)):
        grad_planes.append(ip.Conv2D(src_gray, kernels[i]))
        grad_planes[i] = cv2.convertScaleAbs(grad_planes[i])

    GradImg = cv2.addWeighted(grad_planes[0], 0.5, grad_planes[i], 0.5, 0)
    ip.ImShow("Prewitt Images", GradImg)
    del ip

def Example_ImConv2D_Kirsch():
    ip = cv2IP.ConvIP()
    Img = ip.ImRead(srcImg)    
    ip.ImShow("original", Img)
    src_gray = ip.ImBGR2Gray(Img)
    kernels = ip.GetKirschKernel()
    grad_planes = []
    for i in range(0, len(kernels)):
        grad_planes.append(ip.Conv2D(src_gray, kernels[i]))
    temp_1 = cv2.max(grad_planes[0], grad_planes[1], grad_planes[2])
    temp_2 = cv2.max(grad_planes[3], grad_planes[4], grad_planes[5])
    temp_3 = cv2.max(grad_planes[6], grad_planes[7])
    final = cv2.max(temp_1, temp_2, temp_3)
    ip.ImShow("Kirsch Images", final)
    del ip

def Example_ImSharpening(SpType):
    ip = cv2IP.ConvIP()
    Img = ip.ImRead(srcImg)    
    cv2.imshow("original", Img)
    DstImg = ip.ImSharpening(Img, SpType)
    cv2.imshow("result", DstImg)
    del ip

if __name__ == '__main__':
    srcImg = "C:\\VSCode\\Python\\OpenCV-Python--main\\PyCV2IP\\imgs\\ref.jpg"
    refImg = "C:\\VSCode\\Python\\OpenCV-Python--main\\PyCV2IP\\imgs\\src.jpg"
    # srcImg = "C:\\VSCode\\OpenCV\\PyCV2IP\\imgs\\nature.jpg"
    BackGround = "C:\\VSCode\\Python\\OpenCV\\PyCV2IP\\imgs\\img03.jpg"
    # Example_AlphaBlend()
    Title = "Original Image"
    EQ_Title = "Image Color Equalized"
    # Example_ColorHistEqualize_Original(CType=cv2IP.ColorType.USE_YUV)
    # Example_ImEdge(EdType=cv2IP.EdgeType(2))
    # Example_ImSharpening(SpType=cv2IP.SharpType(4))
    # Example_Smooth(smType=cv2IP.SmoothType(5))
    Example_ImConv2D_Kirsch()
    # Example_ImConv2D_Prewitt()
    cv2.waitKey(0)