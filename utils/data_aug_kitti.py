# this code is used for [image,mask,...] group augmentation, 
# DO support multi-image-group data
# for example, you can process groups like[img,img2,mask,mask2] 
# by modifying the input params and corresponding codes
# all based on the albumentation, numpy and opencv module

import os
import time
import random
import cv2
from cv2 import cv2 as cv
import numpy as np
from albumentations import RandomBrightness,RandomContrast

def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1


def _maybe_process_in_chunks(process_fn, **kwargs):

    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                chunk = img[:, :, index : index + 4]
                chunk = process_fn(chunk, **kwargs)
                chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def aug_rotate(data, saveto, name,
    angle_limit=30, scale_limit=0.3, shift_limit=0.3, interpolation=cv.INTER_LINEAR, \
        border_mode=cv.BORDER_CONSTANT, write_image=True):

    #get data
    img = data['img']
    pc = data['pc']
    lane = data['lane']
    road = data['road']

    #get params
    angle = random.uniform(-angle_limit, angle_limit)
    scale = random.uniform(1.0-scale_limit, 1.0+scale_limit)
    dx = random.uniform(-shift_limit, shift_limit)
    dy = random.uniform(-shift_limit, shift_limit)

    height, width = lane.shape
    center = (width / 2, height / 2)
    matrix = cv.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    warp_affine_fn = _maybe_process_in_chunks(
        cv.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, \
            borderMode=border_mode, borderValue=None
    )
    img = warp_affine_fn(img)
    pc = warp_affine_fn(pc)
    lane = warp_affine_fn(lane)
    road = warp_affine_fn(road)

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_rotate_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_rotate_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_rotate_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_rotate_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'lane': lane, 'road': road}


def hflip(img):
    return np.ascontiguousarray(img[:, ::-1, ...])


def hflip_cv(img):
    return cv.flip(img, 1)


def aug_flip(data, saveto, name, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    lane = data['lane']
    road = data['road']

    img = hflip_cv(img)
    pc = hflip_cv(pc)
    lane = hflip(lane)
    road = hflip(road)

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_flip_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_flip_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_flip_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_flip_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'lane': lane, 'road': road}


def aug_perspective(data, saveto, name, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    lane = data['lane']
    road = data['road']

    #get params
    h, w = lane.shape
    h1,w1 = np.random.randint(50,int(h/3)), np.random.randint(100,int(w/4))
    h2,w2 = np.random.randint(h-int(h/3),h-50), np.random.randint(100,int(w/4))
    h3,w3 = np.random.randint(50,int(h/3)), np.random.randint(w-int(w/4),w-100)
    h4,w4 = np.random.randint(h-int(h/3),h-50), np.random.randint(w-int(w/4),w-100)
    pts1 = np.float32([[w1, h1], [w2, h2], [w4, h4], [w3, h3]])
    pts2 = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
    M = cv.getPerspectiveTransform(pts1, pts2)

    img = cv.warpPerspective(img, M, (w,h))
    pc = cv.warpPerspective(pc, M, (w,h))
    lane = cv.warpPerspective(lane, M, (w,h))
    road = cv.warpPerspective(road, M, (w,h))

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_perspective_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_perspective_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_perspective_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_perspective_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'lane': lane, 'road': road}


def downscale(img, scale, interpolation=cv.INTER_NEAREST):
    h, w = img.shape[:2]
    need_cast = interpolation != cv.INTER_NEAREST and img.dtype == np.uint8
    if need_cast:
        img = to_float(img)
    downscaled = cv.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    upscaled = cv.resize(downscaled, (w, h), interpolation=interpolation)
    if need_cast:
        upscaled = from_float(np.clip(upscaled, 0, 1), dtype=np.dtype("uint8"))
    return upscaled


def gauss_noise(image, var_limit=(30.0, 50.0), mean=0):
    image = image.astype("float32")
    var = random.uniform(var_limit[0], var_limit[1])
    sigma = var ** 0.5
    random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
    gauss = random_state.normal(mean, sigma, image.shape)
    return image + gauss


def aug_noise(data, saveto, name, scale_min=0.25, scale_max=0.5, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    lane = data['lane']
    road = data['road']

    #add noise
    img = gauss_noise(img)
    if np.random.randint(0,2):
        img = downscale(img, scale=random.uniform(scale_min,scale_max))
    else:
        pc = downscale(pc, scale=random.uniform(scale_min,scale_max))

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_noise_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_noise_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_noise_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_noise_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'lane': lane, 'road': road}


def crop_array(data,h1,h2,w1,w2,channels):
    result = np.zeros(data.shape)
    if channels>1:
        result[h1:h2,w1:w2,:] = data[h1:h2,w1:w2,:]
    else:
        result[h1:h2,w1:w2] = data[h1:h2,w1:w2]
    return result


def aug_crop(data, saveto, name, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    lane = data['lane']
    road = data['road']

    #get crop coordinate
    height_limit, width_limit = pc.shape[0], pc.shape[1] 
    h_crop_min = np.random.randint(100, height_limit-150)
    h_crop_max = np.random.randint(max(h_crop_min+100,200), height_limit)
    w_crop_min = np.random.randint(0, width_limit-500)
    w_crop_max = np.random.randint(w_crop_min+300, width_limit)

    img = crop_array(img, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 3)
    pc = crop_array(pc, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 3)
    lane = crop_array(lane, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 1)
    road = crop_array(road, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 1)

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_crop_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_crop_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_crop_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_crop_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'lane': lane, 'road': road}


def aug_brightness(data, saveto, name, write_image=True):
    #get data
    img = data['img'].astype('uint8')
    pc = data['pc']
    lane = data['lane']
    road = data['road']

    aug1 = RandomBrightness(limit=0.4, p=1)
    aug2 = RandomContrast(p=1)
    img = aug1(image=img)['image']
    img = aug2(image=img)['image']

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_brightness_contrast_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_brightness_contrast_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_brightness_contrast_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_brightness_contrast_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'lane': lane, 'road': road}


def aug_lane_erase(data, saveto, name, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    lane = data['lane']
    road = data['road']
    
    #get params from lane label
    original_height, original_width = lane.shape
    X, Y = np.nonzero(lane)
    index = [(X[i],Y[i]) for i in range(len(X)) \
        if X[i] in range(200,original_height-50) and \
            Y[i] in range(100,original_width-50)]
    if len(index)>1: # ensure the label is not empty
        x,y = index[np.random.randint(1,len(index))]
        #agjust exposure based on images 
        img = img.astype(np.float32)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        if gray.mean()<0.5: #dark img
            color = np.random.randint(0,50)
        else: #light img
            color = np.random.randint(200,255)
        radius = np.random.randint(10,50)

        for ch in range(0,3):
            img[x-radius:x+radius, y-radius:y+radius, ch] = color

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_erase_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_erase_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_erase_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_erase_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'lane': lane, 'road': road}


def aug_compose(data, saveto, name, write_image=True):
    # random lose modal with 10% probability
    p = np.random.randint(0,10)
    img, pc = data['img'], data['pc']
    if p==0:
        #img = data['img']
        img = np.zeros(img.shape)
        #data['img'] = img
    else:
        #pc = data['pc']
        pc = np.zeros(pc.shape)
        #data['pc'] = pc
    flag = 0
    if np.random.randint(0,2):
        _data = aug_flip(data=data,saveto=saveto,name=name,write_image=False)
        flag = 1
    if np.random.randint(0,2):
        _data = aug_noise(data=data,saveto=saveto,name=name,write_image=False)
        flag = 1
    if np.random.randint(0,2):
        _data = aug_brightness(data=data,saveto=saveto,name=name,write_image=False)
        flag = 1
    if np.random.randint(0,2):
        _data = aug_crop(data=data,saveto=saveto,name=name,write_image=False)
        flag = 1
    if np.random.randint(0,2):
        _data = aug_lane_erase(data=data,saveto=saveto,name=name,write_image=False)
        flag = 1
    if np.random.randint(0,2):
        _data = aug_rotate(data=data,saveto=saveto,name=name,write_image=False)
        flag = 1
    if np.random.randint(0,2):
        _data = aug_perspective(data=data,saveto=saveto,name=name,write_image=False)
        flag = 1

    if flag:
        img = _data['img']
        pc = _data['pc']
        lane = _data['lane']
        road = _data['road']
    else:
        lane = data['lane']
        road = data['road']

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_compose_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_compose_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_compose_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_compose_road.png'),road)
    else:
        return data

def aug_drop(data, modal, saveto, name, write_image=True):
    '''
    drop modality
    '''

    img = data['img']
    pc = data['pc']
    lane = data['lane']
    road = data['road']

    if modal=='image':
        #img = data['img']
        img = np.zeros(img.shape)
        #data['img'] = img
    elif modal=='point':
        #pc = data['pc']
        pc = np.zeros(pc.shape)
        #data['pc'] = pc

    if write_image:
        cv.imwrite(os.path.join(saveto,'train_image_2_lane',name+'_aug_drop_img.png'),img)
        cv.imwrite(os.path.join(saveto,'knn_pc',name+'_aug_drop_pc.png'),pc)
        cv.imwrite(os.path.join(saveto,'lane_label',name+'_aug_drop_lane.png'),lane)
        cv.imwrite(os.path.join(saveto,'road_label',name+'_aug_drop_road.png'),road)
    else:
        return data

def aug_batch(path_lst, saveto_path):
    PATH = [] #img,pc,lane,road

    for path in path_lst: #rank files
        files = os.listdir(path)
        files.sort()
        PATH.append(files)
    tmp_PATH = np.array(PATH)
    print(tmp_PATH.shape)
    i = 0
    for img,pc,lane,road in zip(PATH[0],PATH[1],PATH[2],PATH[3]):
        i+=1
        if i<300:
            saveto = saveto_path+'train/'
        else:
            saveto = saveto_path+'val/'
        print('-'*20+'processing data '+str(i)+'/383'+'-'*20)
        print(img,pc,lane,road)
        if os.path.exists(os.path.join('./aug/train_image_2_lane/',img[:-4],'_pc_aug_drop_img.png')):
            print('exist')
            pass
        else:
            start_time = time.perf_counter()
            data_name = img[:-4]
            img_data = cv.imread(path_lst[0]+img)
            pc_data = cv.imread(path_lst[1]+pc)
            lane_data = cv.imread(path_lst[2]+lane,cv.IMREAD_GRAYSCALE)
            road_data = cv.imread(path_lst[3]+road,cv.IMREAD_GRAYSCALE)

            data = {'img': img_data, 'pc': pc_data, \
                    'lane': lane_data, 'road': road_data}
            aug_rotate(data=data,saveto=saveto, name=data_name+'_0')
            #aug_rotate(data=data,saveto=saveto, name=data_name+'_1')

            aug_flip(data=data,saveto=saveto, name=data_name+'_0')
            #aug_flip(data=data,saveto=saveto, name=data_name+'_1')

            aug_perspective(data=data,saveto=saveto, name=data_name+'_0')

            aug_noise(data=data,saveto=saveto, name=data_name+'_0')
            #aug_noise(data=data,saveto=saveto, name=data_name+'_1')

            aug_crop(data=data,saveto=saveto, name=data_name+'_0')

            aug_brightness(data=data,saveto=saveto, name=data_name+'_0')
            #aug_brightness(data=data,saveto=saveto, name=data_name+'_1')

            #aug_lane_erase(data=data,saveto=saveto, name=data_name+'_0')
            #aug_lane_erase(data=data,saveto=saveto, name=data_name+'_1')

            aug_compose(data=data, saveto=saveto, name=data_name+'_0')
            aug_compose(data=data, saveto=saveto, name=data_name+'_1')
            aug_compose(data=data, saveto=saveto, name=data_name+'_2')
            aug_compose(data=data, saveto=saveto, name=data_name+'_3')
            aug_compose(data=data, saveto=saveto, name=data_name+'_4')
            '''
            aug_compose(data=data, saveto=saveto, name=data_name+'_5')
            aug_compose(data=data, saveto=saveto, name=data_name+'_6')
            aug_compose(data=data, saveto=saveto, name=data_name+'_7')
            aug_compose(data=data, saveto=saveto, name=data_name+'_8')
            '''
            aug_drop(data=data, modal='image', saveto=saveto, name=data_name+'_img')
            aug_drop(data=data, modal='point', saveto=saveto, name=data_name+'_pc')
            print('time used:%.3f'%(time.perf_counter()-start_time),'seconds')
            #break


if __name__ == "__main__":
    
    path_img = './train_image_2_lane/'
    path_pc = './knn_pc/'
    path_lane = './lane_label/'
    path_road = './road_label/'
    path_lst = [path_img, path_pc, path_lane, path_road]
    saveto = './aug/'
    aug_batch(path_lst=path_lst, saveto_path=saveto)


