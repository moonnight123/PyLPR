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
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
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

kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

image = cv2.dilate(image, kernelX)
image = cv2.erode(image, kernelX)

image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)

image = cv2.medianBlur(image, 21)

contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if (weight > (height * 3.5)) and (weight < (height * 4)):
        image = origin_image[y:y + height, x:x + weight]

gray_image = gray_guss(image)
# 图像阈值化操作——获得二值化图
ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
plt_show(image)

#膨胀操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
image = cv2.dilate(image, kernel)
plt_show(image)

contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
words = []
word_images = []
#对所有轮廓逐一操作
for item in contours:
    word = []
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    word.append(x)
    word.append(y)
    word.append(weight)
    word.append(height)
    words.append(word)
# 排序，车牌号有顺序。words是一个嵌套列表
words = sorted(words,key=lambda s:s[0],reverse=False)
i = 0
#word中存放轮廓的起始点和宽高
for word in words:
    # 筛选字符的轮廓
    if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 3.5)) and (word[2] > 25):
        i = i+1
        splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
        word_images.append(splite_image)
        print(i)
print(words)

for i,j in enumerate(word_images):
    plt.subplot(1,7,i+1)
    plt.imshow(word_images[i],cmap='gray')
plt.show()