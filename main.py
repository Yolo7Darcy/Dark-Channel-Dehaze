from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2
import math
import operator;


#return the dark channel martrix with [:,:]
def DCget(image, radius ):
    heigth, width = image.shape[:-1]
    gray = np.zeros(image.shape,dtype=np.uint8)
    for i in range(heigth):
        for j in range(width):
            for c in range(3):
                gray[i,j,c] = np.min(image[i,j,:])
    kernel = np.ones([radius,radius],dtype=np.uint8)
    gray = cv2.erode(gray, kernel)
    gray = gray[:, :,0]
    return gray

# return the min of 3 channel divide A
def DCget_(image, a, radius ):
    # A = a[0, 0] + a[0, 1] + a[0, 2]
    # A /= 3
    heigth, width = image.shape[:-1]
    gray = np.zeros(image.shape[:-1], dtype=np.float)
    for i in range(heigth):
        for j in range(width):
            sad = image[i, j, 0]/a[0, 0]
            if image[i, j, 1]/a[0, 1] < sad:
                sad = image[i, j, 1]/a[0, 1]
            if image[i, j, 2] / a[0, 2] < sad:
                sad = image[i, j, 2] / a[0, 2]
            gray[i, j] = sad

    # gray = gray / A
    kernel = np.ones([radius, radius], dtype=np.uint8)
    gray = cv2.erode(gray, kernel)
    return gray


#return the atmospheric value
def Airlight(image, dark,Arate ,dark_yuzhi = 220):
    heigth, width = image.shape[:-1]

    size = heigth * width
    npx = int(math.floor(size * Arate))
    if npx < 1:
        npx = 1
    darklist = dark.reshape(size,1)
    imglist = image.reshape(size,3)

    darklist = dark.reshape(size,1)
    imglist = image.reshape(size,3)

    darklist = darklist[:,0]
    index = darklist.argsort()
    index = index[size - npx:,]#默认升序，删掉前面的较小值
    atmsum = np.zeros([1,3])
    for i in range(npx):
        atmsum = atmsum + imglist[index[i]]
    ###应该之记录最大值
    A = atmsum / npx
    for i in range(3):
        print(A[0, 1])
        if A[0, i] > dark_yuzhi:
            A[0, i] = dark_yuzhi

    return A

def TransmissionMat(dark, atm, dark_1, w):
    A = np.sum(atm)
    A /= 3
    heigth, width = dark.shape
    for i in range(heigth):
        for j in range(width):
            dark[i, j] = (1 - w * dark[i, j])
            # temp = dark_1[i,j]
            # B = A - temp
            #
            # #abs
            # if B < 0:
            #     B = -B
            #
            # if B - 0.3137254901960784 < 0.0000000000001:
            #     dark[i, j] = (1 - w * dark[i, j]) * (0.3137254901960784 / (B));
            # else:
            #     dark[i, j] = 1 - w * dark[i, j]
            #
            # if dark[i, j] <= 0.2:
            #     dark[i, j]= 0.5;
            #
            # if dark[i, j] >= 1:
            #     dark[i, j] = 1.0;

    return dark


def dehaze(image, t, A, exposure = 0, t0 = 0.1):
    AAA = np.max(A)
    heigth, width = image.shape[:-1]
    deimg = image.copy()
    for i in range(heigth):
        for j in range(width):
            q = t[i, j]
            if q < t0:
                q = t0
            # q = max(q, t0)
            for c in range(3):
                deimg[i, j, c] = (deimg[i, j, c] - AAA) / q + AAA + exposure

    return deimg

# convert the image to grayscale image
def gray(image):
    gray = np.zeros(image.shape[:-1])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray[i, j] = np.mean(image[i, j])

    return gray

# 参数的设定
w = 0.95#w为大雾的去除率，如果为1的话，会使图像丧失远近感，所以在这里取95%
Radius = 10#radius表示进行最小值滤波以及暗通道处理时的
Arate = 0.001#表示在选择大气光照值的时候取图像像素殿降序排列的前Arate个
t0 = 0.1#在计算透射值t时，发现有一些情况下t的值比较小，这样的图像整体偏白，所以设定一个t的最小值
dark_yuzhi = 200


pil_im = Image.open('3.png')
im = np.array(pil_im)
# im = im.astype('float64') / 255;

dark_channel = DCget(im, Radius)#计算暗通道
A = Airlight(im,dark_channel, Arate, dark_yuzhi)#计算大气光照值
dark = DCget_(im, A,Radius)
te = TransmissionMat(dark, A, dark_channel, w)#计算透射值
j = dehaze(im, te, A,0)#图像还原（去雾）


# cv2.imshow("dark_channel", dark_channel);
# cv2.imshow("dark", dark);
# cv2.imshow("t", j);
# cv2.imshow('I', im);
# cv2.imwrite("J.png", j);
# cv2.waitKey()

plt.subplot(1,4,1)
plt.imshow(im)
plt.subplot(1,4,3)
plt.imshow(dark_channel)
plt.subplot(1,4,4)
plt.imshow(te)
plt.subplot(1,4,2)
plt.imshow(j)
plt.show()
