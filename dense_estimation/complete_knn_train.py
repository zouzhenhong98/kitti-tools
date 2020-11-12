# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:57:23 2020

@author: kerui
"""

# 将一个点云的值赋值为离他最近的三个有值点的加权值
import numpy as np
import cv2
import kdtree
import data_provider
import config
import velo_2_cam
import os
import glob
import cv2
import ransac
import time

def complete_knn(data_dir, image_path, img_size, pc):
    # 创建灰度图像
    img_gray = np.zeros(img_size, dtype=np.uint8)
    height_img_gray = np.zeros(img_size, dtype=np.uint8)
    depth_img_gray = np.zeros(img_size, dtype=np.uint8)
    start1 = time.clock()
    # 先将点云放到二维图像对应的坐标位置
    for i in range(np.size(pc, 1)):
        (x,y) = (int(pc[1, i]), int(pc[0, i]))
        if img_gray[x,y] == 0:
            # reflectance
            img_gray[x,y] = int(pc[3, i]*255)
            # height
            height_img_gray[x,y] = int((pc[2, i]+3)*85) # +3 to ensure positive; 255/3=85 to normalize
            # depth 3D
            depth_img_gray[x,y] = int(pc[5, i]*3) # 255/80->3 to normalize
    start2 = time.clock()    
    # 将图像二维坐标点放进kdtree中
    # 生成二维坐标
    x = range(0, img_size[0])
    y = range(0,img_size[1])
    X, Y = np.meshgrid(x, y) # 生成二维网格坐标
    X, Y = X.flatten(), Y.flatten()
    coordinate = [[x,y] for x, y in zip(X, Y)]
    
    # 有值点的坐标
    value_coordinate = [[x, y] for x, y in zip(X, Y) if img_gray[x, y] > 0]
    
    # 构造KDTree
    KNN = kdtree.create(value_coordinate)
    start3 = time.clock()

    # 存放最近三个点的坐标与权重的字典
    dictionary = {}
    for x, y in coordinate:
        # 如果该点有值
        if [x, y] in value_coordinate:
            continue
        else:
            a1, b1, c1= KNN.search_knn([x, y], 3)
            # 获取最近的三个点的坐标
            a = a1[0].data
            b = b1[0].data
            c = c1[0].data
            # 获取最近的三个点距离当前点的距离
            da = a1[1]
            db = b1[1]
            dc = c1[1]
            # 计算权重
            d_s = da + db + dc
            wa, wb, wc = da/d_s, db/d_s, dc/d_s
    			
            # 将最近三个点的坐标与权重存到字典中
            dictionary[(x, y)] = (a, b, c, wa, wb, wc)
    
    start4 = time.clock()
    # pointcloud completion
    for i, j in coordinate:
            if(img_gray[i, j] == 0):
                a, b, c, da_, db_, dc_ = dictionary[(i,j)]
                A = img_gray[tuple(a)]
                B = img_gray[tuple(b)]
                C = img_gray[tuple(c)]
                img_gray[i,j] = da_*A + db_*B + dc_ *C # ABC are reflectance on the 3 points    
                
                A = height_img_gray[tuple(a)]
                B = height_img_gray[tuple(b)]
                C = height_img_gray[tuple(c)]
                height_img_gray[i,j] = da_*A + db_*B + dc_ *C # ABC are height on the 3 points    

                A = depth_img_gray[tuple(a)]
                B = depth_img_gray[tuple(b)]
                C = depth_img_gray[tuple(c)]
                depth_img_gray[i,j] = da_*A + db_*B + dc_ *C # ABC are depth on the 3 points          

    start5 = time.clock()			
    # generating merged data
    merge = cv2.merge([img_gray, height_img_gray, depth_img_gray])
    cv2.imwrite(data_dir+'/'+'knn_pc_reflectance_height'+'/'+ image_path[45:], merge)
    
    elapsed = []
    for i in [start1,start2,start3]:
        elapsed.append(time.clock() - i)
    print("Time used:",elapsed)
						

def completion(data_dir, path_pc, path_image, path_calib):
    i = 0
    for image_path, pc_path, calib_path in zip(path_image, path_pc, path_calib):
        i+=1
        print('-'*30, 'generating %d data' %i, '-'*30)
        param = data_provider.read_calib(calib_path, [2,4,5])

        # 读取二进制点云文件
        lidar = data_provider.read_pc2array(pc_path, 
                                        height=[-2,-1], #[-1.75,-1.55]
                                        font=True)
        lidar = np.array(lidar)
        lidar = ransac.ransac(lidar)

        cam2img = param[0].reshape([3,4])   # from camera-view to pixels
        cam2cam = param[1].reshape([3,3])   # rectify camera-view
        vel2cam = param[2].reshape([3,4])   # from lidar-view to camera-view

        image_name = image_path.split('\\')[-1]     
        img = cv2.imread(image_name)
        img_shape = img.shape[:2]
        img_shape_inverse = (img_shape[1],img_shape[0])

        cam_coor, pixel = velo_2_cam.lidar_to_camera_project(trans_mat=vel2cam, 
                                                rec_mat=cam2cam,
                                                cam_mat=cam2img,
                                                data=lidar,
                                                pixel_range=img_shape_inverse
                                                )
        
        complete_knn(data_dir, image_name, img_shape, pixel)
        print(pixel[3].min(),pixel[3].max())

def complete_train(data_dir):
    path_pc = sorted(glob.glob(os.path.join(data_dir, 'velodyne/*.bin')))
    path_image = sorted(glob.glob(os.path.join(data_dir, 'train_image_2_lane/*.png')))
    path_calib = sorted(glob.glob(os.path.join(data_dir, 'train_calib/*.txt')))
    completion(data_dir, path_pc, path_image, path_calib)


def complete_test(data_dir):
    path_pc = sorted(glob.glob(os.path.join(data_dir, 'velodyne/*.bin')))
    path_image = sorted(glob.glob(os.path.join(data_dir, 'test_image_2_lane/*.png')))
    path_calib = sorted(glob.glob(os.path.join(data_dir, 'test_calib/*.txt')))
    completion(data_dir, path_pc, path_image, path_calib)
        

if __name__ == "__main__":
    start = time.clock()

    train_dir = '../data_provider/training'
    test_dir = '../data_provider/testing'
    complete_train(train_dir)
    #complete_test(test_dir)

    elapsed = (time.clock() - start)
    print("Time used:",elapsed)


