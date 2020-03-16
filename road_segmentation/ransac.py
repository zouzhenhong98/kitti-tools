# extract the road area with ransac on point clouds

import sys
sys.path.append("..")
import numpy as np
from utils import data_provider
from utils import velo_2_cam
from utils import config
import cv2
import pcl 

def ransac(points_):
    points = points_[:3].T
    cloud = pcl.PointCloud()
    cloud.from_array(points)

    print('Point cloud data: ' + str(cloud.size) + ' points')
    for i in range(0, cloud.size):
        print('x: ' + str(cloud[i][0]) + ', y : ' +
              str(cloud[i][1]) + ', z : ' + str(cloud[i][2]))

    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.06)    # threshold
    seg.set_normal_distance_weight(0.1)
    seg.set_max_iterations(1000)
    indices, coefficients = seg.segment()

    if len(indices) == 0:
        print('Could not estimate a planar model for the given dataset.')
        exit(0)

    print('Model inliers: ' + str(len(indices)))
    return points_[:,indices]

if __name__ == "__main__":
    
    filename = "um_000000"
    pc_path = "../data/bin/"+filename+".bin"
    calib_path = "../data/calib/"+filename+".txt"
    image_path = "../data/img/"+filename+".png"
    print('using data ',filename,' for test')

    # loar filtered pointcloud
    lidar = data_provider.read_pc2array(pc_path, 
                                        height=[-2,-1], # centeal height=-1.7m
                                        font=True)
    lidar = np.array(lidar)
    lidar = ransac(lidar)   # filter with RANSAC
    print('\nfiltered pointcloud size: ', (np.size(lidar,1), np.size(lidar,0)))
    param = data_provider.read_calib(calib_path, [2,4,5])

    # projection: pixels = cam2img * cam2cam * vel2cam * pointcloud
    # matrix type: np.array
    cam2img = param[0].reshape([3,4])   # from camera-view to pixels
    cam2cam = param[1].reshape([3,3])   # rectify camera-view
    vel2cam = param[2].reshape([3,4])   # from lidar-view to camera-view

    HRES = config.HRES          # horizontal resolution (assuming 20Hz setting)
    VRES = config.VRES          # vertical res
    VFOV = config.VFOV          # Field of view (-ve, +ve) along vertical axis
    Y_FUDGE = config.Y_FUDGE    # y fudge factor for velodyne HDL 64E
    
    # get camera-view coordinates & pixel coordinates(after cam2img)
    cam_coor, pixel = velo_2_cam.lidar_to_camera_project(trans_mat=vel2cam, 
                                                        rec_mat=cam2cam, 
                                                        cam_mat=cam2img, 
                                                        data=lidar, 
                                                        pixel_range=(1242,375)
                                                        )
    
    # project pixels to figure
    velo_2_cam.show_pixels(coor=pixel, saveto="../result/ransac_"+filename+".png")

    # add pixels to image
    velo_2_cam.add_pc_to_img(img_path=image_path, coor=pixel, saveto='../result/ransac_'+filename+'_composition2.png')