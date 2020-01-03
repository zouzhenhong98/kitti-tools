This is a package special for kitti data, temporally only support road-detetction channel data.

data_provider:
	read lidar and image data.

show_lidar:
	project lidar pointclouds to panoramic images directly.

velo_2_img:
	transform lidar pointclouds to camera-view image, and save as .png.

folder data:
    save image, pointcoulds and calibrate file.

folder result:
    save result.

TODO:
- [x] load data as .pcd format
- [ ] add pcl processing operation
- [x] create folder and load image via opencv
- [x] add projected lidat pixels to image
- [ ] add lidar segmentation operation
