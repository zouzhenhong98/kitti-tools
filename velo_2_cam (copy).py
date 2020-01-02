import matplotlib.pyplot as plt
import numpy as np
import mayavi.mlab

'''
function:
    (1) project lidar 
numpy anglr is mearsured by pi rather than 360
TODO: 
    calibrate the lidar image with camera image
    fit the dpi argument
'''
def lidar_to_2d_front_view(points,
                           rows,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

        Supported image type: depth, height, reflectence

    Args:
        points: (np array)
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

    '''
    x_lidar = points[:rows, 0]
    y_lidar = points[:rows, 1]
    z_lidar = points[:rows, 2]
    r_lidar = points[:rows, 3] # Reflectance
    '''
    rows = np.where([(points[:rows, 2]<-1.65) & (-1.75<points[:rows, 2])])
    x_lidar = points[rows, 0]
    y_lidar = points[rows, 1]
    z_lidar = points[rows, 2]
    r_lidar = points[rows, 3] # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    D_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES, same as photo on the screen of Velodye
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad # use -y due to Anticlockwise rotation 
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

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

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -D_lidar # originally d_lidar

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



def show_coor2(coor, saveto):
    dpi = 200
    pixel_values = coor[:,2]
    fig,ax = plt.subplots(figsize = (1242/dpi, 375/dpi), dpi = dpi)
    ax.scatter(coor[:,0], -coor[:,1], s=1, c=pixel_values, linewidths=0, alpha=1, cmap='jet')
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)


def lidar_to_camera_project(trans_mat, cam_mat, rec_mat, points, rows):
    '''
    project lidar to camera image
    coor3, pred as input of lidar_to_2d_front_view()
    coor2, directly transformed to image
    '''
    # select height_limited range
    #rows = np.array(np.where(((points[:rows, 2]<-1.65) & (-1.75<points[:rows,2]))))
    #rows = rows.reshape([-1, np.size(rows)])
    #rows = rows[0]
    rows = np.arange(120000)
    print('\n points', len(rows),np.size(points,0), np.size(points,1))
    tmp = []
    coor3 = []
    coor2 = []

    # velodye to unrectified-cam, use height-filtered coordinates
    for i in rows:
        #index = 
        # if show, [x,y,z]=[-y',-z',x']
        x = (points[i,:3] * trans_mat[0][:3]).sum() + trans_mat[0][3]
        y = (points[i,:3] * trans_mat[1][:3]).sum() + trans_mat[1][3]
        z = (points[i,:3] * trans_mat[2][:3]).sum() + trans_mat[2][3]
        tmp.append([x,y,z])
    tmp = np.array(tmp)
    print('\n tmp',np.size(tmp,0), np.size(tmp,1),tmp)
    # read_pc(points[rows,:])
    
    # unrectified-cam to cam
    for i in range(len(rows)):
        x = (tmp[i,:] * rec_mat[0][:3]).sum()
        y = (tmp[i,:] * rec_mat[1][:3]).sum()
        z = (tmp[i,:] * rec_mat[2][:3]).sum()
        coor3.append([x,y,z,points[i,3]])
    coor3 = np.array(coor3)
    
    # cam to image

    for i in range(len(rows)):
        z = (coor3[i,:3] * cam_mat[2][:3]).sum()
        if (z<0):
            continue
        x = (coor3[i,:3] * cam_mat[0][:3]).sum() / z
        if (x<0 or x>=1242):
            continue
        y = (coor3[i,:3] * cam_mat[1][:3]).sum() / z
        if (y<0 or y>=375):
            continue
        coor2.append([x,y,coor3[i,3]])
    coor2 = np.array(coor2)
    print('\n coor2', np.size(coor2,0), np.size(coor2,1),coor2)
    
    # print height-filtered points
    tmp2 = []
    for i in rows:
        tmp2.append([points[i,0],points[i,1],points[i,2],points[i,3]])
    tmp2 = np.array(tmp2)
   # print('\n tmp2', tmp2[:,2].max())

    return coor3, coor2
'''
def read_pc(mat):
    x = mat[:, 2]  # x position of point
    y = -mat[:, 1]  # y position of point
    z = -mat[:, 0]  # z position of point
    r = mat[:, 3]  # reflectance value of point

    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    r_min = r.min()
    r_dis = r.max() - r.min()
    r_ = (r - r_min) / r_dis
    r_ = np.array([i**2 for i in r_]) * 255
    # return x,y,z,r_

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z, r_,          # Values used for Color
                        mode="point",
                        colormap='spectral', # 'bone', 'copper', 'gnuplot'
                        #color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                        figure=fig,
                        )
    
    x=np.linspace(5,5,50)
    y=np.linspace(0,0,50)
    z=np.linspace(0,5,50)
    mayavi.mlab.plot3d(x,y,z)
    mayavi.mlab.show()
'''

if __name__ == "__main__":
    filename = "um_000000"
    
    lidar = np.fromfile(filename+".bin", dtype=np.float32, count=-1).reshape([-1,4])

    rows = 120000
    HRES = 0.35         # horizontal resolution (assuming 20Hz setting)
    VRES = 0.4          # vertical res
    VFOV = (-24.9, 2.0) # Field of view (-ve, +ve) along vertical axis
    Y_FUDGE = 5         # y fudge factor for velodyne HDL 64E
    '''
    matrix = []
    with open(filename+".txt") as f:
        for content in f.readlines()[2,4,5]:
            digits = [i for i in content if str.isdigits(i)]
            matrix.append("".join(digits))
    '''
    proj_cam = [
    [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 
    4.485728000000e+01],
    [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01], 
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]]
    # the fourth column is the movement to cam0, now cam2

    tr_vel_2_cam = [7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03, 1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02, 9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01]

    rec_cam = [
    [9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03],[-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03], [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01],
    ]

    y = lambda mat: np.array(mat).reshape([3,4]).tolist()

    tr_vel_2_cam = y(tr_vel_2_cam)

    lidar, lidar_img = lidar_to_camera_project(tr_vel_2_cam, proj_cam, rec_cam, lidar, rows)

    # direct-trans image
    show_coor2(lidar_img, "./velo2img_"+filename+".png")

    # two-stage-trans image
    # lidar_to_2d_front_view(lidar, rows, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth", saveto="./lidar_depth.png", y_fudge=Y_FUDGE)

    # lidar_to_2d_front_view(lidar, rows, v_res=VRES, h_res=HRES, v_fov=VFOV, val="height", saveto="./lidar_height.png", y_fudge=Y_FUDGE)

    # lidar_to_2d_front_view(lidar, rows, v_res=VRES, h_res=HRES, v_fov=VFOV, val="reflectance", saveto="./lidar_reflectance.png", y_fudge=Y_FUDGE)