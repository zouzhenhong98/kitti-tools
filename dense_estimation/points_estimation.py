'''
point clouds estimation: transfer sparse map to dense map,
work for both depth and reflectance.
'''
import sys
sys.path.append("..")
from utils import data_provider
from utils import velo_2_cam
import numpy as np

# fetch image and point clouds: coordinates and reflectance
def rawData(pc_path_, img_path_):
    
    # loar filtered pointcloud
    lidar = data_provider.read_pc2array(pc_path_, height=None, font=True)
    lidar = np.array(lidar)
    print('\nfiltered pointcloud size: ', (np.size(lidar,1), np.size(lidar,0)))
    # load image
    img = data_provider.read_img(img_path_)
    
    return img, lidar


# project points on the image plane
def lidarPreprocess(point_cloud_, calib_path_, type_):
    
    # type_: r:reflectance, 2d:2d depth, 3d:3d_depth
    assert type_ in {"r", "2d", "3d"}, \
        "type_ should be r:reflectance or 2d:2d_depth or 3d:3d_depth"

    param = data_provider.read_calib(calib_path_, [2,4,5])

    # projection: pixels = cam2img * cam2cam * vel2cam * pointcloud
    # matrix type: np.array
    cam2img = param[0].reshape([3,4])   # from camera-view to pixels
    cam2cam = param[1].reshape([3,3])   # rectify camera-view
    vel2cam = param[2].reshape([3,4])   # from lidar-view to camera-view

    # get camera-view coordinates & pixel coordinates(after cam2img)
    __, pixel = velo_2_cam.lidar_to_camera_project(trans_mat=vel2cam, 
                                                        rec_mat=cam2cam, 
                                                        cam_mat=cam2img, 
                                                        data=point_cloud_, 
                                                        pixel_range=(1242,375)
                                                        )

    if type_=="r":
        pixel = np.row_stack((pixel[:2,:],pixel[3,:]))
        print("return 2d coodinates with reflectance")
    elif type_=="2d":
        pixel = np.row_stack((pixel[:2,:],pixel[4,:]))
        print("return 2d coodinates with 2d depth")
    elif type_=="3d":
        pixel = np.row_stack((pixel[:2,:],pixel[5,:]))
        print("return 2d coodinates with 3d depth")

    return pixel


def completion(point_cloud_):
    """codes wait for completion"""
    pass



if __name__ == "__main__":
    
    filename = "um_000000"
    pc_path = "../data/bin/"+filename+".bin"
    calib_path = "../data/calib/"+filename+".txt"
    image_path = "../data/img/"+filename+".png"
    print('using data ',filename,' for test')

    img, lidar = rawData(pc_path_=pc_path, img_path_=image_path)
    pixel = lidarPreprocess(point_cloud_=lidar, 
                            calib_path_=calib_path, type_="r")
    
    # add pixels to image
    # velo_2_cam.add_pc_to_img(img_path=image_path, coor=pixel, saveto='./result/'+filename+'_composition.png')