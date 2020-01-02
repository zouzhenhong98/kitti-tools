import numpy as np
import pcl
import pypcd

# read calibrate parameters, return P2, R0, tr_vel_2_cam
def read_calib(filename: str, line: list):    
    '''
    parameter:
        filename: path to load data
        line: which line in the file to load
    '''
    with open(filename) as f:
        mat = []
        content = f.readlines()
        for i in line:
            # filter data, get transform matrix
            matrix = content[i].split(' ')[1:]
            name = content[i].split(' ')[0][:-1]
            print('load params: ', name, ', size: ', len(matrix))
            mat.append(np.array([i.strip("\n") for i in matrix], dtype='float32'))
    return mat


# read pointcloud data in .bin file, return python array
def read_pointcloud(filename: str):
    #pointcloud = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
    #p = pcl.PointCloud(pointcloud)
    pc = pypcd.PointCloud.from_path(filename)
    return p,pc


# load lidar pointcloud and project in 3-axis space using mayavi, return list
def read_pc2array(filename: str, 
            height=None, # tuple
            font=None    # bool
            ):
    
    '''
    parameters:
        filename: 
            path to load data
        height: 
            the value indicates whether filtering from height, the tuple contains value of max_height and min_height, like[min, max]
        font: 
            the value indicates whether filtering for font-view only
    output:
        [x, y, z, r, dist2, dist3], type: list[np.array]
    '''

    pointcloud = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
    
    # get the min-length of the four columns[x,y,z,r]
    rows = min([np.size(pointcloud[:,i]) for i in range(np.size(pointcloud,1))])
    print("\npointcloud.shape: ",pointcloud.shape)

    # get the index filtered by min-length, height and font-view
    rows = np.arange(rows)
    if height != None:
        z = pointcloud[:,2]
        filter_h = np.where(((z>=height[0]) & (z<=height[1])))
        rows = rows[np.in1d(rows, filter_h, assume_unique=True)]
    if font != None:
        filter_f = np.where(pointcloud[:,0]>=0)
        rows = rows[np.in1d(rows, filter_f, assume_unique=True)]

    x = pointcloud[rows, 0]  # x position of point
    y = pointcloud[rows, 1]  # y position of point
    z = pointcloud[rows, 2]  # z position of point
    r = pointcloud[rows, 3]  # reflectance value of point

    dist2 = np.sqrt(x**2 + y**2)       # Map distance from sensor, point-wise
    dist3 = np.sqrt(dist2**2 + z**2)   # Space distance from sensor

    print('\nreturn [x, y, z, r, dist2, dist3]')
    return [x, y, z, r, dist2, dist3]


# read image, return python array
def read_img(filename: str):
    pass



if __name__ == "__main__":
    #print('for test\n', read_calib('data/calib/um_000000.txt', [2,4,5]))
    read_pointcloud('data/bin/um_000000.bin')
