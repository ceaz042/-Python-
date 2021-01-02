from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import cv2
import sys
import cv2IP

# def Example_ColorHistEqualize():
#     def click_event(event, x, y, flags, param):
#         global sign
#         if event == cv2.EVENT_LBUTTONDOWN:
#             sign += 1
#             if sign >3:
#                 sign = 1
#             F_eq = Hist.ColorEqualize(img, CType=cv2IP.ColorType(sign))
#             Hist.ImWindow(EQ_Title)
#             Hist.ImShow(EQ_Title, F_eq)
#             Feq_Hist = Hist.CalcColorHist(F_eq)
#             Hist.ShowColorHist("Foreground Color Equalized Hist", Feq_Hist)
#         if event == cv2.EVENT_RBUTTONDBLCLK:
#             sign -= 1
#             if sign <1:
#                 sign = 3
#             F_eq = Hist.ColorEqualize(img, CType=cv2IP.ColorType(sign))
#             Hist.ImWindow(EQ_Title)
#             Hist.ImShow(EQ_Title, F_eq)
#             Feq_Hist = Hist.CalcColorHist(F_eq)
#             Hist.ShowColorHist("Foreground Color Equalized Hist", Feq_Hist)
#     global sign
#     sign = 1
#     Hist = cv2IP.HistIP()
#     img = Hist.ImRead(srcImg)
#     # img = cv2.resize(img, (640, 480))
#     while True:        
#         cv2.setMouseCallback(Title, click_event)
#         Hist.ImWindow(Title)
#         Hist.ImShow(Title, img)
#         F_Hist = Hist.CalcColorHist(img)
#         Hist.ShowColorHist("Foreground Color Hist", F_Hist)        
#         key = cv2.waitKey(30)
#         if key == ord('q') or key == 27:
#             del Hist
#             break

# def Mid_Project():
#     def click_event(event, x, y, flags, param):
#         global sign
#         if event == cv2.EVENT_LBUTTONDOWN:
#             sign += 1
#             if sign >3:
#                 sign = 1
#             outImg = Hist.HistMatching(src_img, ref_img, CType=cv2IP.ColorType(sign))
#             Hist.ImShow("out img", outImg)
#             out_Hist = Hist.CalcColorHist(outImg)
#             out_Hist_cdf = out_Hist.cumsum()
#             Hist.ShowColorHist("Hist after matching", out_Hist_cdf)

#         if event == cv2.EVENT_RBUTTONDBLCLK:
#             sign -= 1
#             if sign <1:
#                 sign = 3
#             outImg = Hist.HistMatching(src_img, ref_img, CType=cv2IP.ColorType(sign))
#             outImg = Hist.HistMatching(src_img, ref_img, CType=cv2IP.ColorType(sign))
#             Hist.ImShow("out img", outImg)
#             out_Hist = Hist.CalcColorHist(outImg)
#             out_Hist_cdf = out_Hist.cumsum()
#             Hist.ShowColorHist("Hist after matching", out_Hist_cdf)
#     global sign
#     sign = 1
#     Hist = cv2IP.HistIP()
#     src_img = Hist.ImRead(srcImg)
#     ref_img = Hist.ImRead(refImg)
#     while True:        
#         cv2.setMouseCallback(Title, click_event)
#         Hist.ImShow("ref img", ref_img)
#         Hist.ImShow(Title, src_img)
#         O_Hist = Hist.CalcColorHist(src_img)
#         Hist.ShowColorHist("Original Color Hist", O_Hist)     
#         key = cv2.waitKey(30)
#         if key == ord('q') or key == 27:
#             del Hist
#             break

class Application(object):
    def __init__(self, master):
        style = ttk.Style()
        # print(style.theme_names())
        # style.configure("Label", foreground="white", background="#242424", bd=1, width=500, height=220)
        style.theme_use('xpnative')
        self.rootframe = ttk.Frame(master, relief="sunken")
        self.rootframe.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.buttomline2 = ttk.Frame(master)
        self.rootframe.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.buttomline3 = ttk.Frame(master)
        self.rootframe.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=2, pady=2)
        self.buttomline4 = ttk.Frame(master)
        self.scale = Frame(master)
        self.rootframe.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)
        self.rootframe.pack(side='top', fill='x')
        self.buttomline2.pack(side='top', fill='x')
        self.buttomline3.pack(side='top', fill='x')
        self.buttomline4.pack(side='top', fill='x')
        self.scale.pack(side='top', fill='x')
        self.setupUI()
        self.__Title = "Original Image"
        self.__EQ_Title = "Image Color Equalized"
        self.__SMTitle = "Smoothed Image"
        self.__EDTitle = "Edge Detected Image"
        self.__UMTitle = "Sharpened Image"
        
    def setupUI(self):
        # ttk.Label(self.rootframe).pack()        
        self.button1 = ttk.Button(self.rootframe)
        self.pathlabel = Label(self.rootframe)
        self.button2 = ttk.Button(self.buttomline2)
        self.button3 = ttk.Button(self.buttomline2)
        self.button4 = ttk.Button(self.buttomline3)
        self.button5 = ttk.Button(self.buttomline3)
        self.button6 = ttk.Button(self.buttomline3)
        self.button7 = ttk.Button(self.buttomline4)
        self.scale_kernel = Scale(self.scale, from_=1, to=15, tickinterval=4, orient="horizontal", resolution=1)
        self.scale_gain = Scale(self.scale, from_=0.05, to=0.95, tickinterval=0.2, orient="horizontal", resolution=0.05)
        self.button1["text"] = "請選擇圖片"        
        self.button1["command"] = self.pick_image
        self.button2["text"] = "進行直方圖等化"        
        self.button2["command"] = self.Example_ColorHistEqualize
        self.button3["text"] = "進行直方圖匹配"        
        self.button3["command"] = self.Hist_Matching
        self.button7["text"] = "離開"        
        self.button7["command"] = self.Exit
        self.button5["text"] = "進行影像平滑"        
        self.button5["command"] = self.Smooth2D
        self.button6["text"] = "進行邊緣偵測"        
        self.button6["command"] = self.EdgeDetect
        self.button4["text"] = "進行影像銳利化"        
        self.button4["command"] = self.ImSharpening
        self.button1.pack(side="left")
        self.pathlabel.pack(side="left")
        self.button2.pack(side="top", fill='x')
        self.button3.pack(side="top", fill='x')
        self.button5.pack(side="top", fill='x')
        self.button6.pack(side="top", fill='x')
        self.button4.pack(side="top", fill='x')
        self.scale_kernel_label = Label(self.scale, text="Kernel").pack(side="top")
        self.scale_kernel.pack(side="top", fill='x')
        self.scale_gain_label = Label(self.scale, text="Gain").pack(side="top")
        self.scale_gain.pack(side="top", fill='x')
        self.button7.pack(side="top", fill='x')
        
    
    def pick_image(self):
        #initialdir 對話框開啟的目錄, title對話框的標題, filetypes找尋的副檔名
        self.img_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"), ("gif files","*.gif"), ("all files","*.*")))
        self.pathlabel.config(text=self.img_path)

    def Example_ColorHistEqualize(self):
        def click_event(event, x, y, flags, param):
            global sign
            if event == cv2.EVENT_LBUTTONDOWN:
                sign += 1
                if sign >3:
                    sign = 1
                F_eq = Hist.ColorEqualize(img, CType=cv2IP.ColorType(sign))
                Hist.ImWindow(self.__EQ_Title)
                Hist.ImShow(self.__EQ_Title, F_eq)
                Feq_Hist = Hist.CalcColorHist(F_eq)
                Hist.ShowColorHist("Image Color Equalized Hist", Feq_Hist)
            if event == cv2.EVENT_RBUTTONDBLCLK:
                sign -= 1
                if sign <1:
                    sign = 3
                F_eq = Hist.ColorEqualize(img, CType=cv2IP.ColorType(sign))
                Hist.ImWindow(self.__EQ_Title)
                Hist.ImShow(self.__EQ_Title, F_eq)
                Feq_Hist = Hist.CalcColorHist(F_eq)
                Hist.ShowColorHist("Image Color Equalized Hist", Feq_Hist)
        global sign
        sign = 1
        Hist = cv2IP.HistIP()
        try:
            img = Hist.ImRead(self.img_path)
        except AttributeError:
            showinfo("錯誤", "請選擇圖片!")
        # img = cv2.resize(img, (640, 480))
        else:
            while True:        
                cv2.setMouseCallback(self.__Title, click_event)
                Hist.ImWindow(self.__Title)
                Hist.ImShow(self.__Title, img)
                F_Hist = Hist.CalcColorHist(img)
                Hist.ShowColorHist("Foreground Color Hist", F_Hist)        
                key = cv2.waitKey(30)
                if key == ord('q') or key == 27:
                    break
    def Hist_Matching(self):
        def click_event(event, x, y, flags, param):
            global sign
            if event == cv2.EVENT_LBUTTONDOWN:
                sign += 1
                if sign >3:
                    sign = 1
                outImg = Hist.HistMatching(src_img, ref_img, CType=cv2IP.ColorType(sign))
                Hist.ImShow("Output image", outImg)
                out_Hist = Hist.CalcColorHist(outImg)

            if event == cv2.EVENT_RBUTTONDBLCLK:
                sign -= 1
                if sign <1:
                    sign = 3
                outImg = Hist.HistMatching(src_img, ref_img, CType=cv2IP.ColorType(sign))
                Hist.ImShow("Output image", outImg)
                out_Hist = Hist.CalcColorHist(outImg)
        global sign
        sign = 1
        Hist = cv2IP.HistIP()
        try:
            src_img = Hist.ImRead(self.img_path)
        except AttributeError:
            showinfo("錯誤", "請選擇圖片!")
        else:
            ref_img_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"), ("gif files","*.gif"), ("all files","*.*")))
            ref_img = Hist.ImRead(ref_img_path)
            while True:        
                cv2.setMouseCallback(self.__Title, click_event)
                Hist.ImShow("ref img", ref_img)
                Hist.ImShow(self.__Title, src_img)
                O_Hist = Hist.CalcColorHist(src_img)
                Hist.ShowColorHist("Original Color Hist", O_Hist)     
                key = cv2.waitKey(30)
                if key == ord('q') or key == 27:
                    break
    
    def Smooth2D(self):
        def click_event(event, x, y, flags, param):
            global sign
            if event == cv2.EVENT_LBUTTONDOWN:
                sign += 1
                if sign >5:
                    sign = 1
                F_SM = CONV.Smooth2D(img, Scale.get(self.scale_kernel), SmType=cv2IP.SmoothType(sign))
                CONV.ImWindow(self.__SMTitle)
                CONV.ImShow(self.__SMTitle, F_SM)
            if event == cv2.EVENT_RBUTTONDBLCLK:
                sign -= 1
                if sign <1:
                    sign = 5
                F_SM = CONV.Smooth2D(img, Scale.get(self.scale_kernel), SmType=cv2IP.SmoothType(sign))
                CONV.ImWindow(self.__SMTitle)
                CONV.ImShow(self.__SMTitle, F_SM)                
        global sign
        sign = 1
        CONV = cv2IP.ConvIP()
        try:
            img = CONV.ImRead(self.img_path)
        except AttributeError:
            showinfo("錯誤", "請選擇圖片!")
        # img = cv2.resize(img, (640, 480))
        else:
            while True:        
                cv2.setMouseCallback(self.__Title, click_event)
                CONV.ImWindow(self.__Title)
                CONV.ImShow(self.__Title, img)    
                key = cv2.waitKey(30)
                if key == ord('q') or key == 27:
                    break

    def EdgeDetect(self):
        def click_event(event, x, y, flags, param):
            global sign
            if event == cv2.EVENT_LBUTTONDOWN:
                sign += 1
                if sign >5:
                    sign = 1
                F_ED = CONV.EdgeDetect(img, EdType=cv2IP.EdgeType(sign))
                CONV.ImWindow(self.__EDTitle)
                CONV.ImShow(self.__EDTitle, F_ED)
            if event == cv2.EVENT_RBUTTONDBLCLK:
                sign -= 1
                if sign <1:
                    sign = 5
                F_ED = CONV.EdgeDetect(img, EdType=cv2IP.EdgeType(sign))
                CONV.ImWindow(self.__EDTitle)
                CONV.ImShow(self.__EDTitle, F_ED)                
        global sign
        sign = 1
        CONV = cv2IP.ConvIP()
        try:
            img = CONV.ImRead(self.img_path)
        except AttributeError:
            showinfo("錯誤", "請選擇圖片!")
        # img = cv2.resize(img, (640, 480))
        else:
            while True:        
                cv2.setMouseCallback(self.__Title, click_event)
                CONV.ImWindow(self.__Title)
                CONV.ImShow(self.__Title, img)    
                key = cv2.waitKey(30)
                if key == ord('q') or key == 27:
                    break

    def ImSharpening(self):
        def click_event(event, x, y, flags, param):
            global sign
            if event == cv2.EVENT_LBUTTONDOWN:
                sign += 1
                if sign >4:
                    sign = 1
                F_UM = CONV.ImSharpening(img, SpType=cv2IP.SharpType(sign), Gain=Scale.get(self.scale_gain), SmType=cv2IP.SmoothType.GAUSSIAN)
                CONV.ImWindow(self.__UMTitle)
                CONV.ImShow(self.__UMTitle, F_UM)
            if event == cv2.EVENT_RBUTTONDBLCLK:
                sign -= 1
                if sign <1:
                    sign = 4
                F_UM = CONV.ImSharpening(img, SpType=cv2IP.SharpType(sign), Gain=Scale.get(self.scale_gain), SmType=cv2IP.SmoothType.GAUSSIAN)
                CONV.ImWindow(self.__UMTitle)
                CONV.ImShow(self.__UMTitle, F_UM)                
        global sign
        sign = 1
        CONV = cv2IP.ConvIP()
        try:
            img = CONV.ImRead(self.img_path)
        except AttributeError:
            showinfo("錯誤", "請選擇圖片!")
        # img = cv2.resize(img, (640, 480))
        else:
            while True:        
                cv2.setMouseCallback(self.__Title, click_event)
                CONV.ImWindow(self.__Title)
                CONV.ImShow(self.__Title, img)    
                key = cv2.waitKey(30)
                if key == ord('q') or key == 27:
                    break

    def Exit(self):
        self.rootframe.quit()

if __name__ == '__main__':
    root = Tk()    
    root.geometry("500x350")
    root.grid_rowconfigure(0, weight=3)
    root.grid_rowconfigure(1, weight=2)
    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=2)
    root.grid_columnconfigure(2, weight=2)    
    app = Application(root)
    # Menu = Label(root, text="hey", width = 30, height = 5)
    # Menu.pack()
    root.mainloop()