import cv2
import numpy as np
import matplotlib.pyplot as mpimg
import matplotlib.pyplot as plt


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


# 读取图像
img = plt.imread('lena.jpg')
img = rgb2gray(img)
print(img.shape)

# 计算线的长度和角度
length = 200
theta = np.pi/4 # 45度的弧度值为pi/4

# 计算线的起点和终点
x1, y1 = 100, 100 # 起点为(100, 100)
x2, y2 = int(x1 + length * np.cos(theta)), int(y1 + length * np.sin(theta))

# 画一条45度的线，线宽为2个像素，颜色为白色
cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

# 显示图像
plt.imshow(img, cmap='gray')
plt.show()