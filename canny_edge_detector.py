import math

from scipy import ndimage
from scipy.ndimage.filters import convolve

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

from utils import utils
import cv2
class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15,std_threshold=9):
        self.imgs = imgs
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        self.std_threshold=std_threshold
        self.max_std=20
        self.std_adjust=0.3
        self.min_intencity_to_adj = 20
        self.max_intencity_to_adj = 100
        return 
    
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        #print(theta)
        return (G, theta)
    

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold;
        lowThreshold = highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self, img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img

    def save(self,im, file):
        with open(file, 'w') as f:
            for i in range(im.shape[0]):
                if (i == 0):
                    for j in range(im.shape[1]):
                        f.write(str(j+1) + '\t')
                    f.write('\n')
                for j in range(im.shape[1]):
                    # if(int(im[i][j]) < 40):
                    # im[i][j]=0;
                    f.write(str(int(abs(im[i][j]))) + '\t')
                f.write('\n')
            for j in range(im.shape[1]):
                f.write(str(j) + '\t')
    def draw_line(self,i,j,thetas,img,gradientMat,nonMaxImg):
       # print(self.gradientMat)
#        # theta = np.pi / 4  # 45度的弧度值为pi/4
       # x1 = 213
       # y1 = 50
        theta = thetas[i,j] + np.pi/2
       # print(math.degrees(theta),"==============")

        # 计算线的长度和角度
        length = 20
        # theta = np.pi / 4  # 45度的弧度值为pi/4
        x1,y1=j,i
        x2, y2 = int(x1 + length * np.sin(theta)), int(y1 + length * np.cos(theta))

        x3, y3 = int(x1 + length * np.sin(theta+np.pi)), int(y1 + length * np.cos(theta+np.pi))
        # 画一条45度的线，线宽为2个像素，颜色为白色
        #cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        #print(x1,y1,x2,y2)
        #cv2.line(img, (x1, y1), (x3, y3), (255, 255, 255), 2)


        # 计算线段上所有点的坐标
        px1s, py1s = np.linspace(x1, x2, num=length, endpoint=True), np.linspace(y1, y2, num=length, endpoint=True)
        points = np.array([np.round(px1s), np.round(py1s)]).T.astype(int)

        # 计算线段上所有点的坐标在这段代码中，使用np.round函数对生成的浮点坐标值进行四舍五入取整，得到的结果是整数坐标。如果计算出的坐标值超出了图像的大小范围，那么在后续的操作中可能会出现越界的情况。因此，在使用这些坐标值进行像素操作之前，通常需要对其进行范围检查和修正，以确保不会出现越界的情况。可以使用类似以下代码对坐标进行修正：
        px2s, py2s = np.linspace(x1, x3, num=length, endpoint=True), np.linspace(y1, y3, num=length, endpoint=True)
        points2 = np.array([np.round(px2s), np.round(py2s)]).T.astype(int)

        points[:, 0] = np.clip(points[:, 0], 0, img.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, img.shape[0] - 1)

        points2[:, 0] = np.clip(points2[:, 0], 0, img.shape[1] - 1)
        points2[:, 1] = np.clip(points2[:, 1], 0, img.shape[0] - 1)

        if(len(points) < length or len(points2) < length ):
            return

        points2_reversed = points2[::-1]
        sum=0
        for k in range(length):
            py1=points[k][0]
            px1=points[k][1]
            py2 = points2[k][0]
            px2 = points2[k][1]
            #print(px1,py1,"   ",px2,py2)
            sum= sum+(self.img_smoothed[px1,py1]- self.img_smoothed[px2,py2])**2
           # print(self.img_smoothed[px1,py1],self.img_smoothed[px2,py2])
        std=np.sqrt(sum)/length
       # print(std)

        if(std<self.std_threshold ):

            adj=self.max_std/(std+1)*self.std_adjust

            test=nonMaxImg[i,j]
            #print(test,"===========",i,j,adj)
            nonMaxImg[i,j] = nonMaxImg[i,j] * adj
           # print(i, j, std, adj, test, nonMaxImg[i, j])
            if (i == 46 and j == 210):
                print("=========46,210======", std)
                print(i,j,std,adj,test,nonMaxImg[i,j])

       # print(std)


       # print(points2)

    def draw_perpendicular_line(self,img,thetas,gradientMat,nonMaxImg):
        # 计算线的长度和角度
        # 计算线的长度和角度

       # self.draw_line(0,0,thetas,img,gradientMat,nonMaxImg)
       # self.draw_line(46, 210, thetas, img, gradientMat, nonMaxImg) #29
       # print(nonMaxImg[46,210])
        #self.draw_line(191, 121, thetas, img, gradientMat, nonMaxImg) #180
       # print(nonMaxImg[191, 121])
        #self.draw_line(206, 63, thetas, img, gradientMat, nonMaxImg)  # 180
        #self.draw_line(15, 428, thetas, img, gradientMat, nonMaxImg)  # 180
       # self.draw_line(19, 493, thetas, img, gradientMat, nonMaxImg)  # 180
       # self.draw_line(497, 378, thetas, img, gradientMat, nonMaxImg)  # 180
      #  print(nonMaxImg[19, 493])

        # 遍历图像中的每一个像素点
        for (i, j), pixel in np.ndenumerate(nonMaxImg):
           # print(i,j,pixel)
            if(pixel > self.min_intencity_to_adj and pixel < self.max_intencity_to_adj):
               #print(i,j,pixel)

                self.draw_line(i, j, thetas, img, gradientMat, nonMaxImg)
                pass
            # i, j 表示像素点的坐标，pixel表示像素点的像素值
            # 这里可以对像素值进行一些处理


       # self.draw_line(212, 50, thetas, img)
        # 显示图像
        plt.imshow(img, cmap='gray')
        plt.show()

    def rgb2gray(self,rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    def detect(self):
        imgs_final = []
        for i, img in enumerate(self.imgs):
           # img = img
           # cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
            #cv2.imshow('image', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            self.save(img, "img.txt")
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.save(self.gradientMat, "gradientMat.txt")
            self.save(self.thetaMat, "thetaMat.txt")



            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            cv2.imwrite('nonMaxImg.jpg',  self.nonMaxImg)
            self.save(self.nonMaxImg, "nonMaxImg.txt")

            self.draw_perpendicular_line(img, self.thetaMat, self.gradientMat,self.nonMaxImg)
            self.save(self.nonMaxImg, "nonMaxImg2.txt")

            print(self.nonMaxImg[46,210],"------------------------")
            self.thresholdImg = self.threshold(self.nonMaxImg)
            cv2.imwrite('thresholdImg.jpg', self.thresholdImg)
            self.save(self.thresholdImg, "thresholdImg.txt")

            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)
            # 读取图像


        return self.imgs_final
