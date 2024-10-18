# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:29:49 2024

@author: ChengZHONG

cv2.imshow reads images in BGR (blue, green, red) order
which is different from plt which reads images in RGB (green, red, blue) order.

"""

"""load the packages"""
#导入相应packages
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from paddleocr import PaddleOCR


#%%
"""show image"""
#显示图片
def show_image(desc, image):
    #cv2.namedWindow(desc, cv2.WINDOW_NORMAL)
    cv2.imshow(desc, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""save image"""
#保存图片
def save_image(filename, image):
    cv2.imwrite('./img/'+filename+'.jpg', image)

#读取图片
img =cv2.imread('./img/car1.jpg')#shapeszie(3656,3656,3)
show_image("img", img)

"""rezie the image"""
# 调整图片大小
lengthsize = 600
widthsize = 600

img = cv2.resize(img, (widthsize, lengthsize))# reszie(img, (widthsize, lengthsize))
show_image("img", img)
save_image('resize', img)

"""Use HSV model to find the blue area"""
# 将rgb模型转化为hsv模型，方便颜色定位
# 根据阈值找到对应颜色

# 蓝色范围
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])
#色相（Hue）、‌饱和度（Saturation）和‌明度（Value）
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#提取蓝色范围的像素点
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
#bitwise_and(src1, src2, mask) 根据src1像素特征位置提取非像素特征位置用作掩码作用与src2
blue = cv2.bitwise_and(hsv, hsv, mask=mask_blue)
show_image('blue', blue)
save_image('blue', blue)

show_image('hsv', hsv)

"""Grayscale the image form 3 tunnels to 1"""
# 灰度图
gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
show_image('gray', gray)
save_image('gray', gray)

"""expansion and corrosion"""
#图片膨胀与腐蚀
element = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
dilation = cv2.dilate(gray, element, iterations=1)
erosion = cv2.erode(dilation, element, iterations=1)
show_image('erosion', erosion)
save_image('erosion', erosion)


#%%
"""found  the coordinates of the vertices"""
# 寻找轮廓（图像矩阵，输出模式，近似方法）
contours, _ = cv2.findContours(erosion.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 根据区域大小排序取前十
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None
# 遍历轮廓，找到车牌轮廓
for c in contours:
 
    # 计算轮廓周长（轮廓，是否闭合）
    peri = cv2.arcLength(c, True)
    # 折线化（轮廓，阈值（越小越接近曲线），是否闭合）返回折线顶点坐标
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # 获取四个顶点（左下/右下/右上/左上)
    if len(approx) == 4:
        screenCnt = approx
        break

"""draw the outline"""
# 如果找到了四边形
if screenCnt is not None:
    # 根据四个顶点坐标对img画线(图像矩阵，轮廓坐标集，轮廓索引，颜色，线条粗细)
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 2)
    show_image('contour', img)
    save_image('contour', img)

#%%
"""mask the non-license plate area"""
# 创建一个灰度图一样大小的图像矩阵
mask = np.zeros(gray.shape, np.uint8)
# 将创建的图像矩阵的车牌区域画成白色
cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
# 图像位运算进行遮罩
mask_image = cv2.bitwise_and(img, img, mask=mask)
show_image('mask_image', mask_image)
save_image('mask_image', mask_image)

#%%
"""crop the license plate area"""
# 获取车牌区域的所有坐标点
(x, y) = np.where(mask == 255)
# 获取底部顶点坐标
(topx, topy) = (np.min(x), np.min(y))
# 获取底部坐标
(bottomx, bottomy,) = (np.max(x), np.max(y))
# 剪裁
cropped = img[topx:bottomx, topy:bottomy]
show_image('cropped', cropped)
save_image('cropped', cropped)

#%%
"""OCR recognition"""
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, ocr_version='PP-OCRv4')
text = ocr.ocr(cropped, cls=True)
for t in text:
    print(t[0][1])
    
