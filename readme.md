### This is a package special for kitti data, 
### including tasks for exploring-data-analysis, sparse-to-dense estimation,
### road segmentation and lane line detection.

-   subfolder and files illustration

-   kitti-tools
├─ data                 [source data]
│  ├─ bin               [lidar point clouds]
│  ├─ calib             [calibration files]
│  ├─ img               [RGB images]
│  ├─ pcd               [generated pcd point clouds]
│  └─ readme.md
├─ dense_estimation     [sparse-to-dense estimation]
│  └─ points_estimation.py
├─ evaluation           [general evalutation code]
│  └─ data_similarity.py
├─ lane_detection       [lane line detection]
├─ requirements.txt
├─ result               [general results]
├─ road_segmentation    [road segmentation]
├─ utils                [general tools]
│  ├─ canny.py
│  ├─ config.py
│  ├─ data_augmentation.py
│  ├─ data_provider.py
│  ├─ show_lidar.py
│  ├─ velo_2_cam.py
│  └─ velo_2_cam_origin.py


TODO:
- [x] load data as .pcd format
- [ ] add pcl processing operation
- [x] create folder and load image via opencv-python
- [x] add projected lidar pixels to image
- [ ] add lidar segmentation operation
- [x] fix modified projection code
- [ ] add lidar estimation code: depth and reflectance
- [x] complete diastance calculation for images
- [x] add canny detecion for points
- [ ] complete data_augmentation code