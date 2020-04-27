# this code is used for [image,mask,...] group augmentation, DO support multi-image-group data
# for example, you can process groups like[img,img2,mask,mask2] by modifying the input params and corresponding codes
# all based on the albumentation, numpy and opencv module


import cv2
import os
import time
import random
import numpy as np
from imgaug import augmenters as iaa

from  albumentations3  import (
    HorizontalFlip, IAAPerspective, CLAHE, RandomRotate90, RandomScale,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, Downscale, RandomSizedCrop, RandomCrop, RandomBrightness
)

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


def aug_rotate(data, saveto, 
    angle_limit=30, scale_limit=0.3, shift_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, write_image=True):

    #get data
    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
    lane = data['lane']
    road = data['road']

    #get params
    angle = random.uniform(-angle_limit, angle_limit)
    scale = random.uniform(1.0-scale_limit, 1.0+scale_limit)
    dx = random.uniform(-shift_limit, shift_limit)
    dy = random.uniform(-shift_limit, shift_limit)

    height, width = lane.shape
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=None
    )
    img = warp_affine_fn(img)
    pc = warp_affine_fn(pc)
    pc2 = warp_affine_fn(pc2)
    lane = warp_affine_fn(lane)
    road = warp_affine_fn(road)

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_rotate_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_rotate_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_rotate_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_rotate_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_rotate_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'pc2': pc2, 'lane': lane, 'road': road}


def hflip(img):
    return np.ascontiguousarray(img[:, ::-1, ...])


def hflip_cv2(img):
    return cv2.flip(img, 1)


def aug_flip(data, saveto, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
    lane = data['lane']
    road = data['road']

    img = hflip_cv2(img)
    pc = hflip_cv2(pc)
    pc2 = hflip(pc2)
    lane = hflip(lane)
    road = hflip(road)

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_flip_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_flip_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_flip_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_flip_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_flip_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'pc2': pc2, 'lane': lane, 'road': road}


def aug_perspective(data, saveto, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
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
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img = cv2.warpPerspective(img, M, (w,h))
    pc = cv2.warpPerspective(pc, M, (w,h))
    pc2 = cv2.warpPerspective(pc2, M, (w,h))
    lane = cv2.warpPerspective(lane, M, (w,h))
    road = cv2.warpPerspective(road, M, (w,h))

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_perspective_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_perspective_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_perspective_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_perspective_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_perspective_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'pc2': pc2, 'lane': lane, 'road': road}


def downscale(img, scale, interpolation=cv2.INTER_NEAREST):
    h, w = img.shape[:2]
    need_cast = interpolation != cv2.INTER_NEAREST and img.dtype == np.uint8
    if need_cast:
        img = to_float(img)
    downscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=interpolation)
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


def aug_noise(data, saveto, scale_min=0.25, scale_max=0.5, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
    lane = data['lane']
    road = data['road']

    #add noise
    img = gauss_noise(img)
    if np.random.randint(0,2):
        img = downscale(img, scale=random.uniform(scale_min,scale_max))

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_noise_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_noise_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_noise_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_noise_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_noise_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'pc2': pc2, 'lane': lane, 'road': road}


def crop_array(data,h1,h2,w1,w2,channels):
    result = np.zeros(data.shape)
    if channels>1:
        result[h1:h2,w1:w2,:] = data[h1:h2,w1:w2,:]
    else:
        result[h1:h2,w1:w2] = data[h1:h2,w1:w2]
    return result


def aug_crop(data, saveto, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
    lane = data['lane']
    road = data['road']

    #get crop coordinate
    height_limit, width_limit = pc2.shape
    h_crop_min = np.random.randint(100, height_limit-150)
    h_crop_max = np.random.randint(max(h_crop_min+100,200), height_limit)
    w_crop_min = np.random.randint(0, width_limit-500)
    w_crop_max = np.random.randint(w_crop_min+300, width_limit)

    img = crop_array(img, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 3)
    pc = crop_array(pc, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 3)
    pc2 = crop_array(pc2, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 1)
    lane = crop_array(lane, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 1)
    road = crop_array(road, h_crop_min, h_crop_max, w_crop_min, w_crop_max, 1)

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_crop_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_crop_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_crop_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_crop_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_crop_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'pc2': pc2, 'lane': lane, 'road': road}


def aug_brightness(data, saveto, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
    lane = data['lane']
    road = data['road']

    aug1 = RandomBrightness(limit=0.5, p=1)
    aug2 = RandomContrast(p=1)
    img = aug1(image=img)['image']
    img = aug2(image=img)['image']

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_brightness_contrast_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_brightness_contrast_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_brightness_contrast_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_brightness_contrast_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_brightness_contrast_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'pc2': pc2, 'lane': lane, 'road': road}


def aug_lane_erase(data, saveto, write_image=True):
    #get data
    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
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
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if gray.mean()<0.5: #dark img
            color = np.random.randint(0,50)
        else: #light img
            color = np.random.randint(200,255)
        radius = np.random.randint(10,50)

        for ch in range(0,3):
            img[x-radius:x+radius, y-radius:y+radius, ch] = color

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_erase_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_erase_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_erase_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_erase_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_erase_road.png'),road)
    else:
        return {'img': img.astype("float32"), 'pc': pc, 'pc2': pc2, 'lane': lane, 'road': road}


def aug_compose(data, saveto, write_image=True):

    if (np.random.randint(0,2)):
        data = aug_flip(data=data,saveto=saveto,write_image=False)
    if (np.random.randint(0,2)):
        data = aug_noise(data=data,saveto=saveto,write_image=False)
    if (np.random.randint(0,2)):
        data = aug_brightness(data=data,saveto=saveto,write_image=False)
    if (np.random.randint(0,2)):
        data = aug_crop(data=data,saveto=saveto,write_image=False)
    if (np.random.randint(0,2)):
        data = aug_lane_erase(data=data, saveto=saveto,write_image=False)
    if (np.random.randint(0,2)):
        data = aug_rotate(data=data,saveto=saveto,write_image=False)
    if (np.random.randint(0,2)):
        data = aug_perspective(data=data,saveto=saveto,write_image=False)
    if (np.random.randint(0,2)):
        data = aug_lane_erase(data=data, saveto=saveto,write_image=False)

    img = data['img']
    pc = data['pc']
    pc2 = data['pc2']
    lane = data['lane']
    road = data['road']

    if write_image:
        cv2.imwrite(os.path.join(saveto,'aug_compose_img.png'),img)
        cv2.imwrite(os.path.join(saveto,'aug_compose_pc.png'),pc)
        cv2.imwrite(os.path.join(saveto,'aug_compose_pc2.png'),pc2)
        cv2.imwrite(os.path.join(saveto,'aug_compose_lane.png'),lane)
        cv2.imwrite(os.path.join(saveto,'aug_compose_road.png'),road)
    else:
        return data


def aug_batch(path_lst, saveto):
    PATH = [] #img,pc,pc2,lane,road

    for path in path_lst: #rank files
        files =os.listdir(path)
        files.sort()
        PATH.append(files)

    for img,pc,pc2,lane,road in zip(PATH[0],PATH[1],PATH[2],PATH[3],PATH[4]):
        img_data = cv2.imread(img)
        pc_data = cv2.imread(pc)
        pc2_data = cv2.imread(pc2,cv2.IMREAD_GRAYSCALE)
        lane_data = cv2.imread(lane,cv2.IMREAD_GRAYSCALE)
        road_data = cv2.imread(road,cv2.IMREAD_GRAYSCALE)

        data = {'img': img_data, 'pc': pc_data, \
                'pc2': pc2_data, 'lane': lane_data, 'road': road_data}
        aug_rotate(data=data,saveto=saveto)
        aug_flip(data=data,saveto=saveto)
        aug_perspective(data=data,saveto=saveto)
        aug_noise(data=data,saveto=saveto)
        aug_crop(data=data,saveto=saveto)
        aug_brightness(data=data,saveto=saveto)
        aug_lane_erase(data=data, saveto=saveto)
        aug_compose(data=data, saveto=saveto)



if __name__ == "__main__":
    img_path = '../data/img/um_000000.png'
    img_path2 = '../data/img/um_000006.png'
    label_path = '../data/label/lane/1.png'
    saveto = '../result/aug_kitti'
    start0 =time.clock()
    img = cv2.imread(img_path)
    pc = cv2.imread(img_path2)
    pc2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    lane = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
    road = cv2.imread(img_path2,cv2.IMREAD_GRAYSCALE)

    # test speed
    data = {'img': img, 'pc': pc, \
            'pc2': pc2, 'lane': lane, 'road': road}
    start1 = time.clock()
    aug_rotate(data=data,saveto=saveto)
    start2 = time.clock()
    aug_flip(data=data,saveto=saveto)
    start3 = time.clock()
    aug_perspective(data=data,saveto=saveto)
    start4 = time.clock()
    aug_noise(data=data,saveto=saveto)
    start5 = time.clock()
    aug_crop(data=data,saveto=saveto)
    start6 = time.clock()
    aug_brightness(data=data,saveto=saveto)
    start7 = time.clock()
    aug_lane_erase(data=data, saveto=saveto)
    start8 = time.clock()
    aug_compose(data=data, saveto=saveto)
    start9 = time.clock()
    print(start9-start8,start8-start7,start7-start6,start6-start5,start5-start4,start4-start3,start3-start2,start2-start1,start1-start0)
    # around 1.3 seconds to conduct process above, using single i5-8250U core