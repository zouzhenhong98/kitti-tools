import numpy as np
from os.path import join
import cv2
import kdtree as KDT
import os
import glob
import time


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def map_lidar_points_onto_image(image_orig, lidar, flag, pixel_size=3):
    image = np.copy(image_orig)
    
    # get rows and cols
    index = [lidar['points'][:,2]<-1.4]
    rows = lidar['row'][index].astype(np.int)
    cols = lidar['col'][index].astype(np.int)

    if flag=='height':
        points_ = lidar['points'][:,2]
    else:  
        points_ = lidar[flag]

    MIN_DISTANCE = np.min(points_)
    MAX_DISTANCE = np.max(points_)
    distances = points_[index]

    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, 1.0, 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = 255 * colours[i]
    return image.astype(np.uint8)


def generate_coor(x_min, x_max, y_min, y_max):
    # generate 2D coor
    x = range(x_min, x_max)
    y = range(y_min, y_max)
    X, Y = np.meshgrid(x, y) # 2D grid
    X, Y = X.flatten(), Y.flatten()
    coordinate = [[x,y] for x, y in zip(X, Y)]
    return (coordinate, x_min, x_max, y_min, y_max)


def complete_knn(reflectance, height, depth, saveto):
    # 创建灰度图像
    start1 = time.clock()

    # Block-wise KNN algorithm
    # 将下半图划分为5*2个区域，从(256,0)(156,100)到(256,400)(156,512)
    # generate blocks to fill
    kd_block = []
    img_size = reflectance.shape
    for i in range(0,1):
        for j in range(0,1):
            if j!=4:
            # scan from the left bottom
                x_max = img_size[0] - i*256
                x_min = x_max - 256
                y_min = j*512
                y_max = y_min + 512
            else:
                x_max = img_size[0] - i*128
                x_min = x_max - 128
                y_min = j*256
                y_max = img_size[1]
            kd_block.append(generate_coor(x_min,x_max,y_min,y_max))
    # the last two blocks on the right edge
    
    # build kd-tree
    start2 = time.clock()
    kdtree_block = []
    for block in kd_block:
        coordinate, x_min, x_max, y_min, y_max = block
        x_value = range(x_min-15, min(x_max+15,img_size[0]))
        y_value = range(max(y_min-15,0), min(y_max+15,img_size[1])) # bounding
        X_value, Y_value = np.meshgrid(x_value, y_value) # 2D grid
        X_value, Y_value = X_value.flatten(), Y_value.flatten()
        # 有值点的坐标
        value_coordinate = [[x, y] for x, y in zip(X_value, Y_value) if reflectance[x, y] > 0]
        if (len(value_coordinate)<20):
            kdtree_block.append((None, None))
        else:
            kdtree = KDT.create(value_coordinate)
            kdtree_block.append((value_coordinate, kdtree))

    start3 = time.clock()
    
    # 存放最近三个点的坐标与权重的字典
    dictionary = {}
    for i in range(len(kd_block)):
        print('start querying %d block' %i)
        coor = kd_block[i][0]  # coordinate
        value_coor, tree = kdtree_block[i]   # value_coordinate, kdtree
        if value_coor is not None:
            for x, y in coor:
                # 如果该点有值
                if [x, y] in value_coor:
                    continue
                else:
                    a1, b1, c1= tree.search_knn([x, y], 3)
                    # 获取最近的三个点的坐标
                    a = a1[0].data
                    b = b1[0].data
                    c = c1[0].data
                    # 获取最近的三个点距离当前点的距离
                    da = a1[1]
                    db = b1[1]
                    dc = c1[1]
                    # 计算权重
                    d_s = da + db + dc
                    wa, wb, wc = da/d_s, db/d_s, dc/d_s

                    # 将最近三个点的坐标与权重存到字典中
                    dictionary[(x, y)] = (a, b, c, wa, wb, wc)

    
    start4 = time.clock()
    # pointcloud completion
    for block in range(len(kd_block)):
        if kdtree_block[block][0] is not None:
            coor_ = kd_block[block]
            for i, j in coor_[0]:
                if(reflectance[i, j] == 0):
                    a, b, c, da_, db_, dc_ = dictionary[(i,j)]
                    A = reflectance[tuple(a)]
                    B = reflectance[tuple(b)]
                    C = reflectance[tuple(c)]
                    reflectance[i,j] = da_*A + db_*B + dc_ *C # ABC are reflectance on the 3 points 

                    A = height[tuple(a)]
                    B = height[tuple(b)]
                    C = height[tuple(c)]
                    height[i,j] = da_*A + db_*B + dc_ *C # ABC are height on the 3 points    

                    A = depth[tuple(a)]
                    B = depth[tuple(b)]
                    C = depth[tuple(c)]
                    depth[i,j] = da_*A + db_*B + dc_ *C # ABC are depth on the 3 points   

    start5 = time.clock()
    # generating merged data
    merge = cv2.merge([reflectance, height, depth])
    cv2.imwrite(saveto, merge)
    
    elapsed = []
    for i in [start1,start2,start3,start4,start5]:
        elapsed.append(time.clock() - i)
    print("Time used:",elapsed)
        

if __name__ == "__main__":

    start = time.clock()
    root_path1 = './20180925_112730/lidar/cam_front_center/' # 4 cores
    root_path2 = './20180925_124435/lidar/cam_front_center/' # 3 cores
    save_path1 = './20180925_112730/result_'
    save_path2 = './20180925_124435/result_'

    file_names = sorted(glob.glob(join(root_path2, '*.npz')))
    for name in file_names:
        print('-'*50,name[-46:])
        if os.path.exists(join(save_path1,name[-46:-4]+'.png')):
            continue
        else:
            lidar_front_center = np.load(name)
            background = np.zeros((1208,1920,3))
            FLAG = ['reflectance','height','depth']
            img_lst = []
            lidar_image = map_lidar_points_onto_image(background, lidar_front_center, flag='reflectance')
            lidar_image = cv2.resize(lidar_image, (512,256), cv2.INTER_AREA)
            #cv2.imwrite(flag_+'1.png',lidar_image)
            #cv2.imwrite(join(save_path2,name[-46:-4]+'.png'),lidar_image)
            lidar_image = cv2.imread(flag_+'1.png',cv2.IMREAD_GRAYSCALE)
            img_lst.append(lidar_image)
            complete_knn(img_lst[0], img_lst[1], img_lst[2], saveto=join('',name[-46:-4]+'.png'))
    print("Time used:",time.clock()-start)

