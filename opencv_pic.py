#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-08-08 16:40:17
# @Author  : Shao Ping (shaooping@163.com)
# @Link    : http://www.shaoping.top
# @Version : $Id$

import os
import cv2
import numpy as np


path = 'img/inpaint.jpg'

img = cv2.imread(path)
hight, width, depth = img.shape[0:3]

#图片二值化处理，把[240, 240, 240]~[255, 255, 255]以外的颜色变成0
thresh = cv2.inRange(img, np.array([240, 240, 240]), np.array([255, 255, 255]))

#创建形状和尺寸的结构元素
kernel = np.ones((3, 3), np.uint8)

#扩张待修复区域
hi_mask = cv2.dilate(thresh, kernel, iterations=1)
specular = cv2.inpaint(img, hi_mask, 5, flags=cv2.INPAINT_TELEA)  # 其中 flags部分可以替换成 cv2.INPAINT_NS，这是两种不同的替换方法
#specular = cv2.inpaint(img, hi_mask, 5, flags=cv2.INPAINT_NS)    #效果并不如INPAINT_TELEA好
 
cv2.namedWindow("Image", 0)
cv2.resizeWindow("Image", int(width / 2), int(hight / 2))
cv2.imshow("Image", img)

cv2.namedWindow("newImage", 0)
cv2.resizeWindow("newImage", int(width / 2), int(hight / 2))
cv2.imshow("newImage", specular)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('img/save.jpg',specular)   #将图片的数组又保存为图片格式，其中specular为新图片的定义名称
cv2.imwrite('img/shandow.jpg',hi_mask)    #查看图片的被掩部分，其中hi_mask为mask的定义名称

# FFFFFF  r:255  g:255  b:255
# 000000  r:0    g:0    b:0                  0~255总共256个颜色值，在rgb三个颜色通道中自由变换，形成了256的三次方种色彩，共计16777216个颜色值。
