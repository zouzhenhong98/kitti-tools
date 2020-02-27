'''
This code is implemented to calculate the similarity between images.
Use cosin distance for image and point clouds 
Use Hash distance + Histogram distance between images
'''
import sys
sys.path.append("..")
from PIL import Image
import cv2
from numpy import average, linalg, dot
 

## cosin distance

def get_thumbnail(image, size=(269, 1242), greyscale=True):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image
 

def image_similarity_vectors_via_numpy(image1, image2):
 
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
 

## hamming distance
'''
import cv2
import numpy as np
from compiler.ast import flatten
import sys

def pHash(imgfile):
    """get image pHash value"""
    #加载并调整图片为32x32灰度图片
    img=cv2.imread(imgfile) 
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)

    #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img       #填充数据

    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32,32)

    #把二维list变成一维list
    img_list=flatten(vis1.tolist()) 

    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]

    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])
'''
'''
# 均值哈希算法
def ahash(image):
    # 将图片缩放为8*8的
    image =  cv2.resize(image, (8,8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # s为像素和初始灰度值，hash_str为哈希值初始值
    s = 0
    ahash_str = ''
    # 遍历像素累加和
    for i in range(8):
        for j in range(8):
            s = s+gray[i, j]
    # 计算像素平均值
    avg = s/64
    # 灰度大于平均值为1相反为0，得到图片的平均哈希值，此时得到的hash值为64位的01字符串
    ahash_str  = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                ahash_str = ahash_str + '1'
            else:
                ahash_str = ahash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(ahash_str[i: i + 4], 2))
    # print("ahash值：",result)
    return result
 
# 差异值哈希算法
def dhash(image):
    # 将图片转化为8*8
    image = cv2.resize(image,(8,8),interpolation=cv2.INTER_CUBIC )
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i, j+1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x'%int(dhash_str[i: i+4],2))
    # print("dhash值",result)
    return result
 
# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n+1
    return n

def useHASH(img1,img2):    
    hash1 = ahash(img1)
    print('img1的ahash值',hash1)
    hash2= dhash(img1)
    print('img1的dhash值',hash2)
    hash3= ahash(img2)
    print('img2的ahash值',hash3)
    hash4= dhash(img2)
    print('img2的dhash值',hash4)
    camphash1 = campHash(hash1, hash3)
    camphash2= campHash(hash2, hash4)
    print("ahash均值哈希差异度：",camphash1)
    print("dhash差异哈希差异度：",camphash2)
'''

 
if __name__ == "__main__":
    
    image1 = Image.open('../data/img/um_000000.png')
    image2 = Image.open('../result/um_000000_composition.png')
    # get Region of Interest(ROI)
    img1 = image1.crop((0, 106, 1242, 375)) # (left, upper, right, lower)
    img2 = image2.crop((0, 106, 1242, 375))

    # cosin distance
    cosin = image_similarity_vectors_via_numpy(img1, img2)
    print(cosin)
    '''
    #ph1 = pHash('./data/img/um_000000.png')
    #print(ph1)
    img1 = cv2.imread('./data/img/um_000000.png')
    img2 = cv2.imread('./result/um_000000_composition.png')
    useHASH(img1,img2)
    '''