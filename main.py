import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
            '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']
chinese_words_list = []
eng_words_list = []
eng_num_words_list = []
theResult = []

win1 = Tk()
win1.title('车牌识别')
win1.geometry('800x500')
filePath = []
global source_img
def jpg_to_gif():
    Path = filedialog.askopenfilename()
    filePath.append(Path)
    global source_img
    im = Image.open(Path)
    im = im.resize((600, 380))
    source_img = ImageTk.PhotoImage(im)
    label = Label(win1,image=source_img)
    label.place(x=50,y=10, anchor=NW)

# plt显示彩色图片
def plt_show0(img):
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

# 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list

# 获得中文模板列表（只匹配车牌的第一个字符）
def get_chinese_words_list():
    for i in range(34,64):
        #将模板存放在字典中
        c_word = read_directory('./refer1/'+ template[i])
        chinese_words_list.append(c_word)
    return chinese_words_list

# 获得英文模板列表（只匹配车牌的第二个字符）
def get_eng_words_list():
    for i in range(10,34):
        e_word = read_directory('./refer1/'+ template[i])
        eng_words_list.append(e_word)
    return eng_words_list

# 获得英文和数字模板列表（匹配车牌后面的字符）
def get_eng_num_words_list():
    for i in range(0,34):
        word = read_directory('./refer1/'+ template[i])
        eng_num_words_list.append(word)
    return eng_num_words_list

# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template,image):
    #将模板进行格式转换
    t_img=cv2.imdecode(np.fromfile(template,dtype=np.uint8),1)
    t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)
    #模板图像阈值化处理——获得黑白图
    ret, t_img = cv2.threshold(t_img, 0, 255, cv2.THRESH_OTSU)
    image_ = image.copy()
    #获得待检测图片的尺寸
    height, width = image_.shape
    # 将模板resize至与图像一样大小
    t_img = cv2.resize(t_img, (width, height))
    # 模板匹配，返回匹配得分
    result = cv2.matchTemplate(image_, t_img, cv2.TM_CCOEFF)
    return result[0][0]


# 对分割得到的字符逐一匹配
def template_matching(word_images):
    results = []
    for index,word_image in enumerate(word_images):
        if index==0:
            best_score = []
            for chinese_words in chinese_words_list:
                score = []
                for chinese_word in chinese_words:
                    result = template_score(chinese_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = template[34+i]
            results.append(r)
            continue
        if index==1:
            best_score = []
            for eng_word_list in eng_words_list:
                score = []
                for eng_word in eng_word_list:
                    result = template_score(eng_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = template[10+i]
            results.append(r)
            continue
        else:
            best_score = []
            for eng_num_word_list in eng_num_words_list:
                score = []
                for eng_num_word in eng_num_word_list:
                    result = template_score(eng_num_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = template[i]
            results.append(r)
            continue
    return results

def FindLRP():
    origin_image = cv2.imread(filePath[-1])
    image = origin_image.copy()
    gray_image = gray_guss(image)
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX

    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=1)

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
        if (weight > (height * 3.5)) and (weight < (height * 5)):
            image = origin_image[y:y + height, x:x + weight]
    gray_image = gray_guss(image)
    # 图像阈值化操作——获得二值化图
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

    # 膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    # 对所有轮廓逐一操作
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
    words = sorted(words, key=lambda s: s[0], reverse=False)
    i = 0
    # word中存放轮廓的起始点和宽高
    for word in words:
        # 筛选字符的轮廓
        if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 3.5)) and (word[2] > 25):
            i = i + 1
            splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            word_images.append(splite_image)

    for i, j in enumerate(word_images):
        plt.subplot(1, 7, i + 1)
        plt.imshow(word_images[i], cmap='gray')

    chinese_words_list = get_chinese_words_list()
    eng_words_list = get_eng_words_list()
    eng_num_words_list = get_eng_num_words_list()

    word_images_ = word_images.copy()
    # 调用函数获得结果
    result = template_matching(word_images_)
    print(result)
    result2 = "".join(result)
    theResult.append(result2)
    print(theResult[0])
    text = Label(win1,text=theResult[-1],font=("Arial", 20))
    text.place(x=150,y=400,anchor=NW)

showImgButton = Button(win1,text="选择图片",command=jpg_to_gif,width=8,height=2)
findCarButton = Button(win1,text="识别车牌号",command=FindLRP,width=10,height=2)
text = Label(win1,text="车牌号为:",font=("Arial", 20))
text.place(x=10,y=400,anchor=NW)
showImgButton.place(x=705,y=40, anchor=W)
findCarButton.place(x=700,y=100, anchor=W)

win1.mainloop()
