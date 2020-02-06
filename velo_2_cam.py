import matplotlib.pyplot as plt
import numpy as np
import mayavi.mlab
import data_provider
import show_lidar
import config
#import pcl
import cv2

#TODO: fix input to lidar_to_2d_front_view()

# project lidar to camera-view image
'''
numpy anglr is mearsured by pi rather than 360
TODO: 
    calibrate the lidar image with camera image
    fit the dpi problem
'''

# directly project pointclouds to 2d font-view without calibration
def lidar_to_2d_front_view(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):

    """ 
    Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

        Supported image type: depth, height, reflectence

    Args:
        points: (np.array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        rows: (integer)
            the rows of points in lidar point cloud
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) == 2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'

    x_lidar = points[0,:]
    y_lidar = points[1,:]
    z_lidar = points[2,:]

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = points[3,:]
    elif val == "height":
        pixel_values = points[4,:]
    else:
        pixel_values = points[5,:]

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES, same as photo on the screen of Velodye
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad # use -y due to Anticlockwise rotation 
    y_img = np.arctan2(z_lidar, points[4,:])/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 200               # Image resolution
    
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    if saveto is not None:
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    else:
        fig.show()



# show pixels via opencv
def show_pixels(coor, saveto, imagepath=None):
    '''load image'''
    img = cv2.imread("")
    dpi = 200
    pixel_values = coor[3,:]
    print(np.shape(coor))

    if imagepath == None:    

        fig,ax = plt.subplots(figsize = (1242/dpi, 375/dpi), dpi = dpi)
        ax.scatter(coor[0,:], -coor[1,:], s=1, c=pixel_values, linewidths=0, alpha=1, cmap='jet')
        ax.set_facecolor((0, 0, 0))    # Set regions with no points to black
        ax.axis('scaled')              # {equal, scaled}
        ax.xaxis.set_visible(False)    # Do not draw axis tick marks
        ax.yaxis.set_visible(False)    # Do not draw axis tick marks
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)

    else:
        show_img(coor=coor,saveto=saveto)


# add projected lidar to image and save
def add_pc_to_img(img_path, coor, saveto=None):
    img = data_provider.read_img(img_path)
    for i in range(np.size(coor,1)):
        img[int(coor[0,i]),int(coor[1,i])] = (0,0,0)  # only int accepted
    if saveto==None:
        cv2.imshow('compose',img)
    else:
        cv2.imwrite(saveto,img)


# project lidar to camera-view coordinates and pixels
def lidar_to_camera_project(trans_mat, 
                            rec_mat, 
                            cam_mat, 
                            data, 
                            pixel_range
                            ):
    
    '''
    parameters:
        trans_mat: from lidar-view to camera-view
        rec_mat: rectify camera-view
        cam_mat: from camera-view to pixelslidar_to_camera_project
        data: use data_provider.read_pc2array() for preprocess
        pixel_range: tuple, the x,y axis range of pixel of image
    output:
        coor: rectified camera-view coordinates
        pixel: pixels projected from rectified camera-view
    '''

    coor = []
    pixel = []
    points = data[:3,:] # coordinates
    value = data[3:,:] # reflectance, dist2, dist3

    '''
    velodye to unrectified-cam, use height-filtered coordinates
    if you want to output coor1/coor2, please reindex [x,y,z]=>[-y',-z',x']
    '''
    coor = np.dot(trans_mat[:,:3], points)
    for i in range(0,3):
        coor[i,:] += trans_mat[i,3]
    coor = np.dot(rec_mat, coor)
    pixel = np.dot(cam_mat[:,:3], coor)
    pixel = pixel[0:2,:] / pixel[2,:].sum() # [u,v] = [x,y] / z

    # reindex coor
    coor = np.array([coor[2], -coor[0], -coor[1]])
    
    # add value to points
    coor = np.row_stack((coor,value))
    pixel = np.row_stack((pixel,value))

    # filter pixel according to image
    x_range = np.where(((pixel[0]>=0) & (pixel[0]<=pixel_range[0])))
    y_range = np.where(((pixel[0]>=0) & (pixel[0]<=pixel_range[1])))
    intersection = np.intersect1d(x_range, y_range)
    pixel = pixel[:,intersection]
    
    # print('\n pixel shape: ', np.shape(pixel))
    # print(coor[0].max(),coor[0].min(),coor[1].max(),coor[1].min())
    # print(pixel[0].max(),pixel[0].min(),pixel[1].max(),pixel[1].min())

    return coor, pixel




if __name__ == "__main__":
    
    filename = "um_000000"
    filepath = "./data/bin/"+filename+".bin"
    parampath = "./data/calib/"+filename+".txt"
    imagepath = "./data/img/"+filename+".png"
    print('using data ',filename,' for test')
    
    '''
    # you can also load pointcloud from bin to pcd
    lidar = data_provider.read_pc2pcd(filepath)
    '''

    # loar filtered pointcloud
    lidar = data_provider.read_pc2array(filepath, 
                                        height=None, #[-1.75,-1.55]
                                        font=True)
    lidar = np.array(lidar)
    print('\nfiltered pointcloud size: ', (np.size(lidar,1), np.size(lidar,0)))
    param = data_provider.read_calib(parampath, [2,4,5])

    '''
    # test pcl
    lidar = lidar[:3,:].T
    p = pcl.PointCloud(lidar)
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    fil.filter().to_file("inliers.pcd")
    '''

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
    cam_coor, pixel = lidar_to_camera_project(trans_mat=vel2cam, 
                                                rec_mat=cam2cam, 
                                                cam_mat=cam2img, 
                                                data=lidar, 
                                                pixel_range=(1242,375)
                                                )

    # project pixels to figure
    show_pixels(coor=pixel, 
                saveto="./result/vel2img_"+filename+".png",
                imagepath=None #"./data/img/"+filename+".png"
                )

    # add pixels to image
    add_pc_to_img('./data/img/'+filename+'.png', coor=pixel, saveto='./result/test.png')
    # from camera-view coordinates project to figure
    
    '''
    # direct projection
    lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, \
        val="depth", saveto="./result/"+filename+"_depth.png", y_fudge=Y_FUDGE)
    
    lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, \
        val="height", saveto="./result/"+filename+"_height.png", y_fudge=Y_FUDGE)

    lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, \
        val="reflectance", saveto="./result/"+filename+"_reflectance.png", y_fudge=Y_FUDGE)
    '''

