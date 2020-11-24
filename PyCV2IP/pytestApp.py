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

class Application(ttk.Frame):
    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.pack()
        self.__Title = "Original Image"
        self.__EQ_Title = "Image Color Equalized"
        self.button1 = ttk.Button(self)
        self.button2 = ttk.Button(self)
        self.button3 = ttk.Button(self)
        self.button4 = ttk.Button(self)
        self.button1["text"] = "請選擇圖片"        
        self.button1["command"] = self.pick_image
        self.button2["text"] = "進行直方圖等化"        
        self.button2["command"] = self.Example_ColorHistEqualize
        self.button3["text"] = "進行直方圖匹配"        
        self.button3["command"] = self.Hist_Matching
        self.button4["text"] = "離開"        
        self.button4["command"] = self.Exit
        self.button1.pack()
        self.button2.pack()
        self.button3.pack()
        self.button4.pack()
    
    def pick_image(self):
        #initialdir 對話框開啟的目錄, title對話框的標題, filetypes找尋的副檔名
        self.img_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"), ("png files","*.png"), ("gif files","*.gif"), ("all files","*.*")))

    def img_hist(self):
        showinfo("錯誤", self.img_path)

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
                # final_Hist = []
                # for i in range(len(out_Hist)):
                #     cdf_Hist = np.cumsum(out_Hist[i])
                #     a = cdf_Hist.tolist()
                #     final_Hist.append(a)
                # final_Hist = np.array(final_Hist)
                # # Hist.ShowColorHist("Hist after matching", out_Hist_cdf)
                # Hist.ShowColorHist("Hist after matching", final_Hist)

            if event == cv2.EVENT_RBUTTONDBLCLK:
                sign -= 1
                if sign <1:
                    sign = 3
                outImg = Hist.HistMatching(src_img, ref_img, CType=cv2IP.ColorType(sign))
                Hist.ImShow("Output image", outImg)
                out_Hist = Hist.CalcColorHist(outImg)
                # final_Hist = []
                # for i in range(len(out_Hist)):
                #     cdf_Hist = np.cumsum(out_Hist[i])

                #     a = cdf_Hist.tolist()
                #     final_Hist.append(a)
                # final_Hist = np.array(final_Hist)                
                # Hist.ShowColorHist("Hist after matching", final_Hist)
                # # Hist.ShowColorHist("Hist after matching", out_Hist)
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
    def Exit(self):
        self.quit()

if __name__ == '__main__':
    root = Tk()    
    app = Application(root)
    # Menu = Label(root, text="hey", width = 30, height = 5)
    # Menu.pack()
    root.mainloop()