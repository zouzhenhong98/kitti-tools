# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:57:23 2020

@author: kerui
"""
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# 将一个点云的值赋值为离他最近的三个有值点的加权值
import numpy as np
import cv2
import kdtree as KDT
import data_provider
#import config
import velo_2_cam
import os
import glob
import cv2
#import ransac
import time
import multiprocessing

def generate_coor(x_min, x_max, y_min, y_max):
    '''generate 2D coor'''
    x = range(x_min, x_max)
    y = range(y_min, y_max)
    X, Y = np.meshgrid(x, y) # 2D grid
    X, Y = X.flatten(), Y.flatten()
    coordinate = [[x,y] for x, y in zip(X, Y)]
    return (coordinate, x_min, x_max, y_min, y_max)


def complete_knn(data_dir, image_path, img_size, pc):
    #print(pc[0].max(),pc[1].max())
    
    start1 = time.perf_counter()
    # create single_channel (gray) images as the blank maps
    intensity_img_gray = np.zeros(img_size, dtype=np.uint8)
    height_img_gray = np.zeros(img_size, dtype=np.uint8)
    depth_img_gray = np.zeros(img_size, dtype=np.uint8)

    # reshape lidar to img_size
    pc[0] = pc[0] * 512 / 1242 # normalize to 0-512
    pc[1] = pc[1] * 256 / 375 # normalize to 0-256
    #print(pc[0].max(),pc[1].max())
    
    # locate the points on the maps

    for i in range(np.size(pc, 1)):
        (x,y) = (int(pc[1, i]), int(pc[0, i]))
        if intensity_img_gray[x,y] == 0:
            # intensity
            intensity_img_gray[x,y] = int(pc[3, i]*255) # times 255 to normalize
            # height
            height_img_gray[x,y] = int((pc[2, i]+3)*85) # +3 to ensure positive; times 255/3=85 to normalize
            # depth 3D
            depth_img_gray[x,y] = int(pc[5, i]*3) # times 255/80->3 to normalize

    
    # generate blocks to fill
    kd_block = []
    for patch in range(0,4):
        x_0 = 256
        x_1 = 200
        x_2 = 140
        x_3 = 0
        y_min = patch * 128
        y_max = y_min + 128
        kd_block.append(generate_coor(x_min=x_1, x_max=x_0, y_min=y_min, y_max=y_max))
        kd_block.append(generate_coor(x_min=x_2, x_max=x_1, y_min=y_min, y_max=y_max))
        kd_block.append(generate_coor(x_min=x_3, x_max=x_2, y_min=y_min, y_max=y_max))

    # build kd-tree
    start2 = time.perf_counter()
    kdtree_block = []
    for block in kd_block:
        coordinate, x_min, x_max, y_min, y_max = block
        x_value = range(max(0,x_min-16), min(x_max+16,256))
        y_value = range(max(y_min-16,0), min(y_max+16,512)) # bounding
        X_value, Y_value = np.meshgrid(x_value, y_value) # 2D grid
        X_value, Y_value = X_value.flatten(), Y_value.flatten()
        # acquire the index of original lidar projection map
        value_coordinate = [[x,y] for x, y in zip(X_value, Y_value) if intensity_img_gray[x,y] > 0]
        if (len(value_coordinate)<10):
            kdtree_block.append((None, None))
        else:
            kdtree = KDT.create(value_coordinate)
            kdtree_block.append((value_coordinate, kdtree))
    start3 = time.perf_counter()
    
    # store the weight and coordinate of 3 nearest points in a dictionary
    neighbor_dict = {}
    for i in range(len(kd_block)):
        #print('start querying %d block' %i)
        coor = kd_block[i][0]  # coordinate
        value_coor, tree = kdtree_block[i]   # value_coordinate, kdtree
        if value_coor is not None: # safety sheck 
            for x, y in coor:
                # query existing points
                if [x, y] in value_coor:
                    continue
                else:
                    a1, b1, c1= tree.search_knn([x, y], 3)
                    # query neighbor points
                    a = a1[0].data
                    b = b1[0].data
                    c = c1[0].data
                    # compute the distance
                    da = a1[1]
                    db = b1[1]
                    dc = c1[1]
                    # compute the weight
                    d_s = da + db + dc
                    wa, wb, wc = da/d_s, db/d_s, dc/d_s
                    # add to neighbor_dict
                    neighbor_dict[(x, y)] = (a, b, c, wa, wb, wc)
                         
                         
    start4 = time.perf_counter()
    # pointcloud completion
    for block in range(len(kd_block)):
        #print('start filling %d block' %block)
        if kdtree_block[block][0] is not None:
            coor_ = kd_block[block]
            for i, j in coor_[0]:
                if(intensity_img_gray[i, j] == 0):
                    a, b, c, da_, db_, dc_ = neighbor_dict[(i,j)]
                    A = intensity_img_gray[tuple(a)]
                    B = intensity_img_gray[tuple(b)]
                    C = intensity_img_gray[tuple(c)]
                    intensity_img_gray[i,j] = da_*A + db_*B + dc_ *C # ABC are intensity on the 3 points 

                    A = height_img_gray[tuple(a)]
                    B = height_img_gray[tuple(b)]
                    C = height_img_gray[tuple(c)]
                    height_img_gray[i,j] = da_*A + db_*B + dc_ *C # ABC are height on the 3 points    

                    A = depth_img_gray[tuple(a)]
                    B = depth_img_gray[tuple(b)]
                    C = depth_img_gray[tuple(c)]
                    depth_img_gray[i,j] = da_*A + db_*B + dc_ *C # ABC are depth on the 3 points  
    
    start5 = time.perf_counter()
    # generating merged data
    merge = cv2.merge([intensity_img_gray, height_img_gray, depth_img_gray])
    cv2.imwrite(data_dir+'velodyne_points/knn_results/'+image_path[-14:], merge)
    
    elapsed = []
    for i in [start1,start2,start3,start4,start5]:
        elapsed.append(time.perf_counter() - i)
    print("Time used:",elapsed)


def completion(data_dir, path_pc, path_image, path_calib):
    '''the main function of completion'''

    '''load the calib file'''
    param = data_provider.read_calib(path_calib, [0,1,2,3])
    cam2img = param[0].reshape([3,4])   # from camera-view to pixels
    cam2cam = param[1].reshape([3,3])   # rectify camera-view
    R = param[2].reshape([3,3])
    T = param[3].reshape([3,-1])
    vel2cam = np.append(R,T,axis=1)     # from lidar-view to camera-view: [R;T] in raw_data
    
    '''project and complete_knn'''
    i = 0
    total_num = len(path_image)
    est_time = 0
    for image_path, pc_path in zip(path_image, path_pc):
        i+=1
        start = time.perf_counter()
        print('-'*30, 'generating %d data of' %i,str(total_num), '-'*30)
        # check the existing completed files
        check_path = os.path.join(data_dir,'velodyne_points/knn_results/',image_path[-14:])
        #print(check_path)
        if os.path.exists(check_path):
            print('pass file {}'.format(image_path))
            total_num -= 1
            continue

        # 读取二进制点云文件
        lidar = data_provider.read_pc2array(pc_path, 
                                        height=[-2,-1], #[-1.75,-1.55]
                                        font=True)
        lidar = np.array(lidar)
        #lidar = ransac.ransac(lidar)

        image_name = image_path.split('\\')[-1]
        img = cv2.imread(image_name)
        img_shape = img.shape[:2] # (375,1242)
        img_shape_inverse = (img_shape[1],img_shape[0])
        cam_coor, pixel = velo_2_cam.lidar_to_camera_project(trans_mat=vel2cam, 
                                                rec_mat=cam2cam,
                                                cam_mat=cam2img,
                                                data=lidar,
                                                pixel_range=img_shape_inverse
                                                )
        # pixel:=[x,y,h,r,d2,d3], x = 0-1242, y = 0-375
        #print(pixel[2].max(),pixel[2].min(),pixel[3].max(),pixel[3].min(),pixel[4].max(),pixel[4].min())
        print(image_name)
        complete_knn(data_dir=data_dir, image_path=image_name, img_size=(256,512), pc=pixel)
        print('estimated time left {}'.format(\
            (time.perf_counter()-start)*(total_num-i)/60))
        #break
        #print(pixel[3].min(),pixel[3].max())



def complete_func(data_dir, calib):
    DIR1 = data_dir[0]
    DIR2 = data_dir[1]
    PATH_pc1 = sorted(glob.glob(os.path.join(DIR1, 'velodyne_points/data/*.bin')))
    PATH_pc2 = sorted(glob.glob(os.path.join(DIR2, 'velodyne_points/data/*.bin')))
    PATH_image1 = sorted(glob.glob(os.path.join(DIR1, 'image_02/data/*.png')))
    PATH_image2 = sorted(glob.glob(os.path.join(DIR2, 'image_02/data/*.png')))
    #completion(data_dir=data_dir, path_pc=PATH_pc, path_image=PATH_image, path_calib=calib)

    # multi-processor should be less than cpu numbers
    # process dataset1
    p11 = multiprocessing.Process(target=completion, args=(DIR1,PATH_pc1[:215],PATH_image1[:215],calib))
    p12 = multiprocessing.Process(target=completion, args=(DIR1,PATH_pc1[215:],PATH_image1[215:],calib))
    # process dataset2
    p21 = multiprocessing.Process(target=completion, args=(DIR2,PATH_pc2[:215],PATH_image2[:215],calib))
    p22 = multiprocessing.Process(target=completion, args=(DIR2,PATH_pc2[215:],PATH_image2[215:],calib))

    p11.start()
    p12.start()
    p21.start()
    p22.start()
        

if __name__ == "__main__":
    start = time.perf_counter()
    data_dir1 = '2011_09_26_drive_0028_sync/'
    data_dir2 = '2011_09_26_drive_0029_sync/'
    calib_file = '2011_09_26/calib.txt' # use the same calib file for all data record at 2011_09_26
    complete_func(data_dir=(data_dir1,data_dir2), calib=calib_file) # data_dir2

    elapsed = (time.perf_counter() - start)
    print("Time used:",elapsed)


