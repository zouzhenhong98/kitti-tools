import numpy as np
import pcl
import cv2
import os

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
    if filename.endswith(".pcd"):
        p = pcl.load(filename)
    elif filename.endswith(".bin"):
        p = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
        p = array2pcd(p)
    else:
        raise ValueError("Only support .bin and .pcd format")

    return p


# numpy to pcd
def array2pcd(points,   # 4xN pointcloud array
            VERSION="0.7",
            FIELDS="x y z reflectence",
            SIZE="4 4 4 4",
            TYPE="F F F F",
            COUNT="1 1 1 1",
            WIDTH=None,
            HEIGHT=None,
            VIEWPOINT="0 0 0 1 0 0 0",
            POINTS=None,    # value same as WIDTH
            DATA="binary",  # binary or ascii
            saveto=None  # save as .pcd
            ):

    '''
    format illustration:
        https://blog.csdn.net/david_xtd/article/details/36898955 
        or 《点云库PCL从入门到精通》
    '''

    if saveto != None:
        
        xlist = points[0].tolist()
        ylist = points[1].tolist()
        zlist = points[2].tolist()
        rlist = points[3].tolist()    # reflectence

        # create file
        if not os.path.exists(saveto):
            f = open(saveto, 'w')
            f.close()
        
        # write info
        with open(saveto, 'w') as file_to_write:
            file_to_write.writelines("# .PCD v0.7 - Point Cloud Data file format\n")
            file_to_write.writelines("VERSION "+VERSION+"\n")
            file_to_write.writelines("FIELDS "+FIELDS+"\n")
            file_to_write.writelines("SIZE "+SIZE+"\n")
            file_to_write.writelines("TYPE "+TYPE+"\n")
            file_to_write.writelines("COUNT "+COUNT+"\n")
            file_to_write.writelines("WIDTH " +str(len(xlist))+"\n")
            file_to_write.writelines("HEIGHT 1\n")
            file_to_write.writelines("VIEWPOINT "+VIEWPOINT+"\n")
            file_to_write.writelines("POINTS "+str(len(xlist))+"\n")
            file_to_write.writelines("DATA "+DATA+"\n")
            for i in range(len(xlist)):
                file_to_write.writelines(str(xlist[i]) + " " + str(ylist[i])\
                     + " " + str(zlist[i]) + " " + str(rlist[i]) + "\n")
            print("\nsuccessfully save to "+saveto)

        # load .pcd
        p = pcl.load(saveto)
        return p
    
    else:
        '''return pcd.pointcloud from np.array, not complete pcd file'''
        p = pcl.PointCloud(points.T)
        print("\nload np.array as pcd format without info")
        return p


# load lidar pointcloud and project in 3-axis space using mayavi, return list
def read_pc2array(filename: str, 
                    height=None, # tuple
                    font=None):
    
    '''
    parameters:
        filename: 
            path to load data
        height: 
            the value indicates whether filtering from height, the tuple\
                 contains value of max_height and min_height, like[min, max]
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
    img = cv2.imread(filename)
    return img


if __name__ == "__main__":
    test_file = 'um_000000'
    # print('for test\n', read_calib('data/calib/'+test_file+'.txt', [2,4,5]))
    # data = read_pc2array('data/bin/'+test_file+'.bin',[-1.75,-1.55],True)
    # read_pointcloud('data/bin/'+test_file+'.bin')
    # p = array2pcd(data, saveto='./data/pcd/'+test_file+'.pcd')
    read_img('./data/img/'+test_file+'.png')

