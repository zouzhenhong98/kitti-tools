import sys
sys.path.append("..")
import numpy as np
from utils import data_provider
from utils import config
from utils import velo_2_cam
import cv2


def lapalian_demo(image):
    #dst = cv.Laplacian(image, cv.CV_32F)#默认为4领域拉普拉斯算子
    #lpls = cv.convertScaleAbs(dst)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])#自定义的8领域拉普拉斯算子
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("lapalian_demo", lpls)



def sobel_demo(image):
    #grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    #grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)#cv.Sobel的增强版，对噪声比较敏感
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x) #求绝对值并转化为8位的图像上
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient-x", gradx)
    cv.imshow("gradient-y", grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient", gradxy)


src = cv.imread("F:/images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
sobel_demo(src)
lapalian_demo(src)
cv.waitKey(0)