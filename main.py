import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


# plt显示彩色图片
def plt_show0(img):
    # cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

# 图像去噪灰度处理
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image


origin_image = cv2.imread('Car.jpg')
image = origin_image.copy()
gray_image = gray_guss(image)

Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(Sobel_x)
image = absX

ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
plt_show(image)
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 1)
plt_show(image)