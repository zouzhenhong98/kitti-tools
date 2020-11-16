import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from __future__ import print_function
import os
import sys
import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_utils as utils
raw_input = input  # Python 3


class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = "E:/KITTI/Dataset/training/image_2"
        self.calib_dir = "E:/KITTI/Dataset/training/calib"
        self.lidar_dir = "E:/KITTI/Dataset/training/velodyne"
        self.label_dir = "E:/KITTI/Dataset/training/label_2"

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
       return imgfov_pc_velo, pts_2d, fov_inds
    else:
       return imgfov_pc_velo


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, \
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        #color = cmap[int(640.0 / depth), :]
        temp = int(640.0 / depth)
        if temp >= 256:
            temp = 255
        color = cmap[temp, :]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i, 0])),
                         int(np.round(imgfov_pts_2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)
    #Image.fromarray(img).show()
    return img


def dataset_viz():
    dataset = kitti_object("E:/KITTI/Dataset/training/image_2")
    i = 0
    while i<len(dataset):
        # Load data from dataset
        objects = dataset.get_label_objects(i)
        objects[0].print_object()
        img = dataset.get_image(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        pc_velo = dataset.get_lidar(i)[:, 0:3]
        calib = dataset.get_calibration(i)
        gray0 = np.zeros((img_height,img_width), dtype=np.uint8)
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        gray0=cv2.cvtColor(gray0, cv2.COLOR_BGR2RGB)
        img=show_lidar_on_image(pc_velo, gray0, calib, img_width, img_height)
        cv2.imshow(img)
        print(str(i) + "finished")
        i += 1

        
if __name__ == '__main__':
    import mayavi.mlab as mlab
    #from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d
    dataset_viz()