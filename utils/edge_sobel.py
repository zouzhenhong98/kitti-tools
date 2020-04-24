import cv2

img = cv2.imread('../result/dense_knn.png')
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
def cv_imshow(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
 
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)  #绝对值转换
cv_imshow(sobelx,'sobelx')
 
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely= cv2.convertScaleAbs(sobely)   #绝对值转换
cv_imshow(sobely,'sobely')
#分别为计算x和y，再求和
sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)  #将x和y融合起来
cv_imshow(sobelxy,'sobelxy')