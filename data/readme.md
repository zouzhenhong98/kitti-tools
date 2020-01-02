Data Illustration:

Calib:
Include calibration parameters for every pointcloud data.

<Each calibration file contains the following matrices (in row-major order):>

	P0 (3x4): Projection matrix for left grayscale camera in rectified coordinates
	P1 (3x4): Projection matrix for right grayscale camera in rectified coordinates
	P2 (3x4): Projection matrix for left color camera in rectified coordinates
	P3 (3x4): Projection matrix for right color camera in rectified coordinates
	R0_rect (3x3): Rotation from non-rectified to rectified camera coordinate system
	Tr_velo_to_cam (3x4): Rigid transformation from Velodyne to (non-rectified) camera coordinates
	Tr_imu_to_velo (3x4): Rigid transformation from IMU to Velodyne coordinates
	Tr_cam_to_road (3x4): Rigid transformation from (non-rectified) camera to road coordinates
	calib_cam_to_cam.txt: Camera-to-camera calibration
	--------------------------------------------------
	- S_xx: 1x2 size of image xx before rectification
	- K_xx: 3x3 calibration matrix of camera xx before rectification
	- D_xx: 1x5 distortion vector of camera xx before rectification
	- R_xx: 3x3 rotation matrix of camera xx (extrinsic)
	- T_xx: 3x1 translation vector of camera xx (extrinsic)
	- S_rect_xx: 1x2 size of image xx after rectification
	- R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
	- P_rect_xx: 3x4 projection matrix after rectification

	**Only P2, R0_rect, Tr_velo_to_cam are utilized in codes.**

Image:

-	1242x375 for pixels.

point clouds:

-	around 120,000 points totally, with 4 parameters: x, y, z, reflectence.
-	lidar parameters:
	HRES = 0.35         # horizontal resolution (assuming 20Hz setting)
    VRES = 0.4          # vertical res
    VFOV = (-24.9, 2.0) # Field of view (-ve, +ve) along vertical axis
    Y_FUDGE = 5         # y fudge factor for velodyne HDL 64E
