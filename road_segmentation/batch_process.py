# extract the road area with ransac on point clouds

import sys
sys.path.append("..")
import numpy as np
from utils import data_provider
from utils import velo_2_cam
from utils import config
from road_segmentation import ransac
import cv2
import pcl
import os
import time
import matplotlib.pyplot as plt

def batchProcess(img_dir, pc_dir, calib_dir, save_to):
    
    indices_ = []
    data = []
    
    # get file indices
    for root, dirs, files in os.walk(img_dir):
        for file_ in files:
            indices_.append(file_[:-4])
    
    # ransac process in batch
    for filename in indices_:

        pc_path = pc_dir+filename+".bin"
        calib_path = calib_dir+filename+".txt"
        image_path = img_dir+filename+".png"

        # loar filtered pointcloud
        lidar = data_provider.read_pc2array(pc_path, 
                                            height=[-2.1,-1.4], # centeal height=-1.7m
                                            font=True)
        lidar = np.array(lidar)
        data.append(ransac.ransac(lidar))

    # # calculate the range of height on the road
    # data = np.array(data)    
    # print(data[:,0].min(),data[:,1].min(),data[:,2].min(),data[:,3].min())
    # fig = plt.figure(figsize=(12, 3))
    # plt.subplot(141)
    # plt.hist(data[:,0],bins=30,color='r')
    # plt.title('max_height')
    # plt.subplot(142)
    # plt.hist(data[:,1],bins=30,color='r')
    # plt.title('min_height')
    # plt.subplot(143)
    # plt.hist(data[:,2],bins=30,color='r')
    # plt.title('range')
    # plt.subplot(144)
    # plt.hist(data[:,3],bins=30,color='r')
    # plt.title('median')
    # plt.show()
    
        lidar = ransac.ransac(lidar)   # filter with RANSAC
        
        param = data_provider.read_calib(calib_path, [2,4,5])

        # projection: pixels = cam2img * cam2cam * vel2cam * pointcloud
        # matrix type: np.array
        cam2img = param[0].reshape([3,4])   # from camera-view to pixels
        cam2cam = param[1].reshape([3,3])   # rectify camera-view
        vel2cam = param[2].reshape([3,4])   # from lidar-view to camera-view

        HRES = config.HRES       # horizontal resolution (assuming 20Hz setting)
        VRES = config.VRES       # vertical res
        VFOV = config.VFOV       # Field of view (-ve, +ve) along vertical axis
        Y_FUDGE = config.Y_FUDGE # y fudge factor for velodyne HDL 64E
        
        # get camera-view coordinates & pixel coordinates(after cam2img)
        cam_coor, pixel = velo_2_cam.lidar_to_camera_project(trans_mat=vel2cam, 
                                                            rec_mat=cam2cam, 
                                                            cam_mat=cam2img, 
                                                            data=lidar, 
                                                            pixel_range=(1242,375)
                                                            )
        
        # project pixels to figure
        velo_2_cam.show_pixels(coor=pixel, saveto=save_to+filename+"_seg.png")
        
        # add pixels to image
        velo_2_cam.add_pc_to_img(img_path=image_path, coor=pixel, saveto=save_to+filename+'_composition.png')



if __name__ == "__main__":

    train_img_dir = '/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road/training/image_2/'
    train_pc_dir = '/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road_velodyne/training/velodyne/'
    train_calib_dir = '/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road/training/train_calib/'
    train_result = '../result/road_seg_ransac/train/'

    test_img_dir = '/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road/testing/image_2/'
    test_pc_dir = '/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road_velodyne/testing/velodyne/'
    test_calib_dir = '/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road/testing/test_calib/'
    test_result = '../result/road_seg_ransac/test/'

    start = time.clock()

    batchProcess(img_dir=train_img_dir, 
                pc_dir=train_pc_dir, 
                calib_dir=train_calib_dir, 
                save_to=train_result)

    batchProcess(img_dir=test_img_dir, 
                pc_dir=test_pc_dir, 
                calib_dir=test_calib_dir, 
                save_to=test_result)

    elapsed = (time.clock() - start)
    print("Time used:",elapsed)
       