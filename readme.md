### This is a package special for kitti data, 
### including tasks for exploring-data-analysis, sparse-to-dense estimation,
### road segmentation and lane line detection.

### to quickly get access to data, please use ./utils/data_provider.py or ./utils/velo_2_cam.py

-   subfolder and files illustration

-   kitti-tools<br/>
├─ data                 [source data]<br/>
│  ├─ bin               [lidar point clouds]<br/>
│  ├─ calib             [calibration files]<br/>
│  ├─ img               [RGB images]<br/>
│  ├─ pcd               [generated pcd point clouds]<br/>
│  └─ readme.md<br/>
├─ dense_estimation     [sparse-to-dense estimation]<br/>
│  └─ points_estimation.py<br/>
├─ evaluation           [general evalutation code]<br/>
│  └─ data_similarity.py<br/>
├─ lane_detection       [lane line detection]<br/>
├─ requirements.txt<br/>
├─ result               [general results]<br/>
├─ road_segmentation    [road segmentation]<br/>
├─ utils                [general tools]<br/>
│  ├─ canny.py<br/>
│  ├─ config.py<br/>
│  ├─ data_augmentation.py<br/>
│  ├─ data_provider.py<br/>
│  ├─ show_lidar.py<br/>
│  ├─ velo_2_cam.py<br/>
│  └─ velo_2_cam_origin.py<br/>


TODO:
- [x] load data as .pcd format
- [ ] add pcl processing operation
- [x] create folder and load image via opencv-python
- [x] add projected lidar pixels to image
- [x] fix modified projection code
- [ ] add lidar estimation code: reflectance
- [x] complete diastance calculation for images
- [x] add canny detecion for points
- [ ] complete data_augmentation code
- [x] add road segmentation with RANSAC on point clouds