#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2 as cv

def imageProcessing(image):
    """
    对图片进行预处理，提取含油区域
    param image: 待处理的图片
    return mask: 经过中值滤波、提取颜色(绿色，黄色)、二值化的图片
    """
    median = cv.medianBlur(image,5) # 增加 50% 噪声，进行中值滤波处理，降噪
    imageHSV = cv.cvtColor(median, cv.COLOR_BGR2HSV) # 转换颜色空间 BGR 到 HSV
    
    # 定义HSV中的范围
    #yellow=np.uint8([[[0,50,0]]]) # RGB
    #hsv_yellow=cv.cvtColor(yellow,cv.COLOR_BGR2HSV) # RGB 转 HSV
    #print(hsv_green[0][0][0],hsv_green[0][0][1],hsv_green[0][0][2]) # 提取 HSV
    lower = np.array([26,43,46]) # 定义HSV中黄色+绿色的范围
    upper = np.array([77,255,255])

    mask = cv.inRange(imageHSV, lower, upper) # 设置HSV的阈值使得只取绿、绿色
    #（所得的mask是二值化的图像，符合颜色的像素值为255，不符合的为0，即表现为图片的亮出和暗处）
    
    # 膨胀去掉小噪点 （未写，待确定其必要性）
    #res = cv.bitwise_and(median,median, mask= mask) # 将掩膜和图像逐像素相加
    #cv.imshow('mask',mask) # 查看图片含油的区域
    return mask

def getArea(mask_):
    """
    获得图片中含油区域的面积
    param mask_: 待计算面积的mask图
    return area: 面积
    """
    area=0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]==255:
                area+=1
    return area


image=cv.imread('C:\\Users\\mi\\Desktop\\1-2.bmp',cv.IMREAD_COLOR) # cv.IMREAD_COLOR =1  以BGR三通道的形式读取图片 
#cv.namedWindow('imageShown',cv.WINDOW_NORMAL) # 查看原图
#cv.imshow('imageShown',image)
results=[]#np.array([[1, 2]])

# 批量读取图片,并进行处理
for i in np.arange(319):
    image = cv.imread('H:\\BaiduNetdiskDownload\\Rock\\'+str(i+1)+'-2.bmp',cv.IMREAD_COLOR) # cv.IMREAD_COLOR =1  以BGR三通道的形式读取图片
    if image is None:
        areaRatio=0
    else:
        mask=imageProcessing(image) # 图片处理
        areaRatio=getArea(mask)/((mask.shape[0])*(mask.shape[1])) #获取含油区域面积
        results.append([i+1,areaRatio])
for j in np.arange(31): #320-350 = 31
    image = cv.imread('H:\\BaiduNetdiskDownload\\Rock\\'+str(j+320)+'-2.jpg',cv.IMREAD_COLOR) # cv.IMREAD_COLOR =1  以BGR三通道的形式读取图片
    if image is None:
        areaRatio=0
    else:
        mask=imageProcessing(image) 
        areaRatio=getArea(mask)/((mask.shape[0])*(mask.shape[1]))
        results.append([j+320,areaRatio])#[j+320,areaRatio]        
#    imageRGB = image[:,:,::-1] # BGR to RGB
#    data=np.zeros((imageRGB.shape[0],imageRGB.shape[1],imageRGB.shape[2]))
#    data[0:image.shape[0],0:image.shape[1]] = imageRGB # 提取得图像的RGB三通道的像素值，存储在矩阵data中

# 储存所有结果
#print(results)
columns = ["样本编号", "含油面积百分含量"]
dt = pd.DataFrame(results, columns=columns)
dt.to_csv("H:\\BaiduNetdiskDownload\\results_2.csv", index=0,encoding='utf_8_sig')

cv.waitKey(0)
cv.destroyAllWindows() #cv.destroyWindow()    

