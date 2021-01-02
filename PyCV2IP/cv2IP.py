import cv2
import numpy as np
import enum
import os

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

class ColorType(enum.IntEnum):
    USE_RGB = 1
    USE_HSV = 2
    USE_YUV = 3

class SmoothType(enum.IntEnum):
    BLUR = 1
    BOX = 2
    GAUSSIAN = 3
    MEDIAN = 4
    BILATERAL = 5

class EdgeType(enum.IntEnum):
    SOBEL = 1
    CANNY = 2
    SCHARR = 3
    LAPLACE = 4
    COLOR_SOBEL = 5

class SharpType(enum.IntEnum):
    LAPLACE_TYPE1 = 1
    LAPLACE_TYPE2 = 2
    SECOND_ORDER_LOG = 3
    UNSHARP_MASK = 4

class HistIP(BaseIP):
    def __init__(self):
        self.__H = 384
        self.__W = 512
        self.__bin_w = int(round( self.__W/256 ))

    def ImBGR2Gray(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)

    def ImBGRA2BGR(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGRA2BGR)

    def CalcGrayHist(self, SrcGray):
        return cv2.calcHist([SrcGray], [0], None, [256], [0, 256])

    def ShowGrayHist(self, Winname, GrayHist):
        Hist = np.zeros((self.__H, self.__W, 3), np.uint8)
        cv2.normalize(GrayHist, GrayHist, alpha=0, beta=self.__H, norm_type=cv2.NORM_MINMAX)
        for i in range(1, 256):
            cv2.line(Hist, ( self.__bin_w*(i-1), self.__H - int(np.round(GrayHist[i-1])) ),
                    ( self.__bin_w*(i), self.__H - int(np.round(GrayHist[i])) ),
                    ( 250, 250, 250), thickness=2)
        BaseIP.ImShow(Winname, Hist)

    def CalcColorHist(self, SrcColor):
        channel = cv2.split(SrcColor)
        Hist = []
        b_hist = cv2.calcHist(channel, [0], None, [256], (0, 256), accumulate=False)
        g_hist = cv2.calcHist(channel, [1], None, [256], (0, 256), accumulate=False)
        r_hist = cv2.calcHist(channel, [2], None, [256], (0, 256), accumulate=False)
        b_hist.tolist()
        g_hist.tolist()
        r_hist.tolist()
        Hist.append(b_hist)
        Hist.append(g_hist)
        Hist.append(r_hist)
        ColorHist = np.array(Hist)
        # print(len(Hist))
        return ColorHist

    def ShowColorHist(self, Winname, ColorHist):
        histImage = np.zeros((self.__H, self.__W, 3), np.uint8)
        # hist = np.array(ColorHist)
        b_hist = ColorHist[0]
        g_hist = ColorHist[1]
        r_hist = ColorHist[2]
        # b_hist = hist[0]
        # g_hist = hist[1]
        # r_hist = hist[2]
        cv2.normalize(b_hist, b_hist, alpha=0, beta=self.__H, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, alpha=0, beta=self.__H, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, alpha=0, beta=self.__H, norm_type=cv2.NORM_MINMAX)
        for i in range(1, 256):
            cv2.line(histImage, ( self.__bin_w*(i-1), self.__H - int(np.round(b_hist[i-1])) ),
                    ( self.__bin_w*(i), self.__H - int(np.round(b_hist[i])) ),
                    ( 255, 132, 0), thickness=2)
            cv2.line(histImage, ( self.__bin_w*(i-1), self.__H - int(np.round(g_hist[i-1])) ),
                    ( self.__bin_w*(i), self.__H - int(np.round(g_hist[i])) ),
                    ( 125, 255, 52), thickness=2)
            cv2.line(histImage, ( self.__bin_w*(i-1), self.__H - int(np.round(r_hist[i-1])) ),
                    ( self.__bin_w*(i), self.__H - int(np.round(r_hist[i])) ),
                    ( 125, 52, 235), thickness=2)
        BaseIP.ImShow(Winname, histImage)

    def MonoEqualize(self, SrcGray):
        return cv2.equalizeHist(SrcGray)
    
    def ColorEqualize(self, SrcColor, CType = ColorType.USE_HSV):
        if CType == ColorType(1):
            Color = cv2.cvtColor(SrcColor, cv2.COLOR_BGRA2BGR)
            print('RGB')
            channel = cv2.split(Color)
            channel_B = cv2.equalizeHist(channel[0])
            channel_G = cv2.equalizeHist(channel[1])
            channel_R = cv2.equalizeHist(channel[2])
            Color = cv2.merge((channel_B, channel_G, channel_R))
            return Color
        if CType == ColorType(2):
            Color = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2HSV)
            print('HSV')
            channel = cv2.split(Color)
            channel_V = cv2.equalizeHist(channel[2])
            Color = cv2.merge((channel[0], channel[1], channel_V))
            Color = cv2.cvtColor(Color, cv2.COLOR_HSV2BGR)
            return Color
        if CType == ColorType(3):
            Color = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2YUV)
            print('YUV')
            channel = cv2.split(Color)
            channel_Y = cv2.equalizeHist(channel[0])
            Color = cv2.merge((channel_Y, channel[1], channel[2]))
            Color = cv2.cvtColor(Color, cv2.COLOR_YUV2BGR)
            return Color
    def HistMatching(self, SrcImg, RefImg, CType = ColorType.USE_HSV):
        def calculate_cdf(Hist):
            pdf = cv2.calcHist([Hist], [0], None, [256], [0, 256])
            # cdf = pdf.cumsum()
            cdf = np.cumsum(pdf)
            # normalized_cdf = cdf*float(pdf.max()/cdf.max())
            normalized_cdf = cdf/float(cdf.max())
            return normalized_cdf

        def calculate_lookup(src_cdf, ref_cdf):
            lookup_table = np.zeros(256)
            lookup_val = 0
            for src_pixel_val in range(len(src_cdf)):
                lookup_val
                for ref_pixel_val in range(len(ref_cdf)):
                    if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                        lookup_val = ref_pixel_val
                        break
                lookup_table[src_pixel_val] = lookup_val
            return lookup_table

        if CType == ColorType(1):
            print('RGB')
            #SrcImg
            src_Color = cv2.cvtColor(SrcImg, cv2.COLOR_BGRA2BGR)
            src_channel = cv2.split(src_Color)
            src_B_channel = src_channel[0]
            src_G_channel = src_channel[1]
            src_R_channel = src_channel[2]
            src_cdf_B = calculate_cdf(src_B_channel)
            src_cdf_G = calculate_cdf(src_G_channel)
            src_cdf_R = calculate_cdf(src_R_channel)
            #RefImg
            ref_Color = cv2.cvtColor(RefImg, cv2.COLOR_BGRA2BGR)
            ref_channel = cv2.split(ref_Color)
            ref_B_channel = ref_channel[0]
            ref_G_channel = ref_channel[1]
            ref_R_channel = ref_channel[2]
            ref_cdf_B = calculate_cdf(ref_B_channel)
            ref_cdf_G = calculate_cdf(ref_G_channel)
            ref_cdf_R = calculate_cdf(ref_R_channel)
            #Calcilate_lookup
            B_lookup_table = calculate_lookup(src_cdf_B, ref_cdf_B)
            G_lookup_table = calculate_lookup(src_cdf_G, ref_cdf_G)
            R_lookup_table = calculate_lookup(src_cdf_R, ref_cdf_R)
            B_after_transform = cv2.LUT(src_B_channel, B_lookup_table)
            G_after_transform = cv2.LUT(src_G_channel, G_lookup_table)
            R_after_transform = cv2.LUT(src_R_channel, R_lookup_table)
            Histogram = []
            result_S = src_cdf_B.tolist()
            result_R = ref_cdf_B.tolist()
            result_Ori = B_after_transform.astype(np.uint8)
            result_Ori = calculate_cdf(result_Ori)
            result_O = result_Ori.tolist()
            Histogram.append(result_S)
            Histogram.append(result_R)
            Histogram.append(result_O)
            Histogram = np.array(Histogram)
            # Histogram = cv2.convertScaleAbs(Histogram)
            self.ShowColorHist("Hist after matching", Histogram)
            img_after_matching = cv2.merge((B_after_transform, G_after_transform, R_after_transform))
            return img_after_matching.astype(np.uint8)

        if CType == ColorType(2):
            print('HSV')
            #SrcImg
            Src_Color = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2HSV)
            src_channel = cv2.split(Src_Color)
            src_H_channel = src_channel[0].astype(float)
            src_S_channel = src_channel[1].astype(float)
            src_V_channel = src_channel[2]
            src_cdf_V = calculate_cdf(src_V_channel)
            #RefImg
            ref_Color = cv2.cvtColor(RefImg, cv2.COLOR_BGR2HSV)
            ref_channel = cv2.split(ref_Color)
            ref_V_channel = ref_channel[2]
            ref_cdf_V = calculate_cdf(ref_V_channel)
            #Calcilate_lookup
            V_lookup_table = calculate_lookup(src_cdf_V, ref_cdf_V)
            V_after_transform = cv2.LUT(src_V_channel, V_lookup_table)
            Histogram = []
            result_S = src_cdf_V.tolist()
            result_R = ref_cdf_V.tolist()
            result_Ori = V_after_transform.astype(np.uint8)
            result_Ori = calculate_cdf(result_Ori)
            result_O = result_Ori.tolist()
            Histogram.append(result_S)
            Histogram.append(result_R)
            Histogram.append(result_O)
            Histogram = np.array(Histogram)
            # Histogram = cv2.convertScaleAbs(Histogram)
            self.ShowColorHist("Hist after matching", Histogram)
            img_after_matching = cv2.merge([src_H_channel, src_S_channel, V_after_transform])
            img_after_matching = img_after_matching.astype(np.uint8)
            img_after_matching = cv2.cvtColor(img_after_matching, cv2.COLOR_HSV2BGR)
            return img_after_matching

        if CType == ColorType(3):
            print('YUV')
            #SrcImg
            Src_Color = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2YUV)
            src_channel = cv2.split(Src_Color)
            src_Y_channel = src_channel[0]
            src_U_channel = src_channel[1].astype(float)
            src_V_channel = src_channel[2].astype(float)
            src_cdf_Y = calculate_cdf(src_Y_channel)
            #RefImg
            ref_Color = cv2.cvtColor(RefImg, cv2.COLOR_BGR2YUV)
            ref_channel = cv2.split(ref_Color)
            ref_Y_channel = ref_channel[0]
            ref_cdf_Y = calculate_cdf(ref_Y_channel)
            #Calcilate_lookup
            Y_lookup_table = calculate_lookup(src_cdf_Y, ref_cdf_Y)
            Y_after_transform = cv2.LUT(src_Y_channel, Y_lookup_table)
            Histogram = []
            result_S = src_cdf_Y.tolist()
            result_R = ref_cdf_Y.tolist()
            result_Ori = Y_after_transform.astype(np.uint8)
            result_Ori = calculate_cdf(result_Ori)
            result_O = result_Ori.tolist()
            Histogram.append(result_S)
            Histogram.append(result_R)
            Histogram.append(result_O)
            Histogram = np.array(Histogram)
            # Histogram = cv2.convertScaleAbs(Histogram)
            self.ShowColorHist("Hist after matching", Histogram)
            img_after_matching = cv2.merge((Y_after_transform, src_U_channel, src_V_channel))
            img_after_matching = img_after_matching.astype(np.uint8)
            img_after_matching = cv2.cvtColor(img_after_matching, cv2.COLOR_YUV2BGR)
            return img_after_matching

class ConvIP(BaseIP):
    def Smooth2D(self, SrcImg, ksize = 15, SmType = SmoothType.BLUR):
        if SmType == SmoothType(1):
            print('BLUR')
            result = cv2.blur(SrcImg, (ksize, ksize))
            return result
        if SmType == SmoothType(2):
            print('BOX')
            result = cv2.boxFilter(SrcImg, -1, (ksize, ksize))
            return result
        if SmType == SmoothType(3):
            print('GAUSSIAN')
            result = cv2.GaussianBlur(SrcImg, (ksize, ksize), 0)
            return result
        if SmType == SmoothType(4):
            print('MEDIAN')
            result = cv2.medianBlur(SrcImg, ksize)
            return result
        if SmType == SmoothType(5):
            print('BILATERAL')
            result = cv2.bilateralFilter(SrcImg, ksize, ksize*2, ksize/2)
            return result
        
    def EdgeDetect(self, SrcImg, EdType = EdgeType.SOBEL):
        if EdType == EdgeType(1):
            print('SOBEL')
            # window_name = ('Sobel Edge Detector')
            ddepth = cv2.CV_16S
            src = self.Smooth2D(SrcImg, 3, SmType= SmoothType.GAUSSIAN)
            gray = HistIP.ImBGR2Gray(self, src)
            grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            return grad
        if EdType == EdgeType(2):
            print('CANNY')
            # window_name = "Canny Edge Detector"
            # max_lowThreshold = 100
            low_threshold = 100
            ratio = 3
            kernel_size = 3
            img_blur = self.Smooth2D(SrcImg, ksize=3, SmType=SmoothType.BLUR)
            detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
            return detected_edges
        if EdType == EdgeType(3):
            print('SCHARR')
            ddepth = cv2.CV_16S
            src = self.Smooth2D(SrcImg, 3, SmType= SmoothType.GAUSSIAN)
            gray = HistIP.ImBGR2Gray(self, src)
            grad_x = cv2.Scharr(gray, ddepth, 1, 0, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Scharr(gray, ddepth, 0, 1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            return grad
        if EdType == EdgeType(4):
            print('LAPLACE')
            ddepth = cv2.CV_16S
            kernel_size = 3
            src = self.Smooth2D(SrcImg, 3, SmType= SmoothType.GAUSSIAN)
            gray = HistIP.ImBGR2Gray(self, src)
            dst = cv2.Laplacian(gray, ddepth, ksize=kernel_size)
            # converting back to uint8
            abs_dst = cv2.convertScaleAbs(dst)
            return abs_dst            
        if EdType == EdgeType(5):
            print('COLOR_SOBEL')
            # window_name = "Color Sobel Edge Detector"
            ddepth = cv2.CV_16S
            src = self.Smooth2D(SrcImg, 3, SmType= SmoothType.GAUSSIAN)
            src_x = cv2.Sobel(src, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            src_y = cv2.Sobel(src, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            abs_src_x = cv2.convertScaleAbs(src_x)
            abs_src_y = cv2.convertScaleAbs(src_y)
            grad = cv2.addWeighted(abs_src_x, 0.5, abs_src_y, 0.5, 0)
            return grad
    def ImSharpening(self, SrcImg, SpType=SharpType.UNSHARP_MASK, Gain=0.5, SmType=SmoothType.GAUSSIAN):
        if SpType == SharpType(1):
            print('LAPLACE_TYPE1')
            Img = SrcImg
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            result = cv2.filter2D(Img, ddepth=-1, kernel=kernel, anchor = (-1, -1), delta = 0, borderType=cv2.BORDER_DEFAULT)
            output = cv2.addWeighted(Img, 1, result, Gain, 0)
            return output

        if SpType == SharpType(2):
            print('LAPLACE_TYPE2')
            Img = SrcImg
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            result = cv2.filter2D(Img, ddepth=-1, kernel=kernel, anchor = (-1, -1), delta = 0, borderType=cv2.BORDER_DEFAULT)
            output = cv2.addWeighted(Img, 1, result, Gain, 0)
            return output

        if SpType == SharpType(3):
            print('SECOND_ORDER_LOG')
            Img = SrcImg
            kernel = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
            result = cv2.filter2D(Img, ddepth=-1, kernel=kernel, anchor = (-1, -1), delta = 0, borderType=cv2.BORDER_DEFAULT)
            output = cv2.addWeighted(Img, 1, result, Gain, 0)
            return output

        if SpType == SharpType(4):
            print('UNSHARP_MASK')
            Img = SrcImg
            smooth = self.Smooth2D(Img, 9, SmType)
            output = cv2.addWeighted(SrcImg, 1+Gain, smooth, Gain*-1, 0)
            return output   