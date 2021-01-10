import os
import cv2
import numpy as np
from random import randint
from random import uniform
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# from keras.preprocessing import image

folder = ''
# folder2 = ''
trainimage_path = ''
# labelimage_path = ''

size_input = 32
size_label = 32
stride = 32

imglist = os.listdir(folder)
# vallist = os.listdir(folder2)
imglist.sort(key=lambda x: int(x[:-4]))
# vallist.sort(key=lambda x:int(x[:-4]))
length = len(imglist)
frames = 7
count = 0
temp = 0
k = 0
l = 0

for i in range(int(length/frames)):
    fn2 = imglist[i]
    imagepath1 = os.path.join(folder, fn2)
    img = cv2.imread(imagepath1, 0)
    [hei, wid] = [img.shape[0], img.shape[1]]
    for x in range(0, hei-size_input+1, stride):
        for y in range(0, wid-size_input+1, stride):
            b = img[x: x+size_input, y: y+size_input]
            [m, n] = b.shape
            for p in range(m):
                for q in range(n):
                    if b[p, q] == 0:
                        l += 1
            if l > 32 * 32 * 8 / 10:
                l = 0
                continue
            for k in range(frames):
                fn = imglist[i * frames + k]
                imagepath2 = os.path.join(folder, fn)
                img1 = cv2.imread(imagepath2)
                img1 = rgb2gray(img1)
                a = img1[x: x+size_input, y: y+size_input]
                imagedata_name = trainimage_path + '/' + str(count) + '.png'
                count += 1
                cv2.imwrite(imagedata_name, a * 255)
