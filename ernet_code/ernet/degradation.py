import math
import os
from random import uniform, randint
import cv2
import numpy as np

folder = ''
save_path = ''
size_input = 32
size_label = 32
stride = 20
ss = 20
withnoise = 1
count = 1
temp = 0
k = 0
c = 1

list = os.listdir(folder)
list.sort()


def filter_atom(M, r0):
    f = 0.1
    d = 1.8
    la = 0.56
    be = 1 / 2
    hm = round(M / 2)
    a = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            x = (i - hm) / M
            y = (j - hm) / M
            U = np.sqrt(x ** 2 + y ** 2)
            a[i, j] = np.exp(-3.44 * ((la*f*U/r0) ** (5 / 3))
                             * (1 - be*(la*f*U/r0)**(1/3)))
            alpha = la * f * U / d
            t = 2 / 3.14 * (math.acos(alpha) - (alpha) *
                            (1 - (alpha ** 2)) ** (1 / 2))
            if(la*f*U <= d):
                a[i, j] = a[i, j] * t
            else:
                a[i, j] = 0
    a = a - a.min()
    a = a / a.max()
    a = a / a.sum()
    h = a
    return h


def motion_blur(image, degree, angle):
    image = np.array(image)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(
        motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


for i in range(len(list)):
    fn = list[i]
    imagepath = os.path.join(folder, fn)
    img = cv2.imread(imagepath, 0)
    a = img
    r0 = 0.015
    l = 15

    for tp in range(7):
        imagedata_name2 = save_path + '/' + str(count) + ".png"
        count += 1
        r0 = uniform(0.010, 0.015)
        l = randint(35, 42)
        th = filter_atom(ss, r0)
        ths = int(np.floor(ss / 2))
        psf = np.zeros((32, 32))
        psf[16 - ths: 16 + ths, 16 - ths: 16 + ths] = th
        b = cv2.filter2D(a, cv2.CV_32FC1, psf)
        noise_Poiss = np.random.poisson(l, size=(img.shape[0], img.shape[1]))
        c = b + np.double(noise_Poiss)
        cv2.imwrite(imagedata_name2, c)
