# this code is used for [image,mask] pair augmentation, do not support multi-image-pair


import cv2
import os
import numpy as np
from  albumentations3  import (
    HorizontalFlip, IAAPerspective, CLAHE, RandomRotate90, RandomScale,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, Downscale, RandomSizedCrop, RandomCrop, RandomBrightness
)

def aug_rotate(param,saveto):
    aug = ShiftScaleRotate(shift_limit=0.3, 
                        scale_limit=0.3, 
                        rotate_limit=30, 
                        border_mode=cv2.BORDER_CONSTANT, 
                        p=1)
    augmented = aug(**param)
    image, mask = augmented["image"], augmented["mask"]
    cv2.imwrite(os.path.join(saveto,'aug_rotate_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_rotate_mask.png'),mask)


def aug_flip(param,saveto):
    aug = HorizontalFlip(p=1)
    augmented = aug(**param)
    image, mask = augmented['image'], augmented['mask']
    cv2.imwrite(os.path.join(saveto,'aug_flip_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_flip_mask.png'),mask)


def aug_perspective(param,saveto):
    aug = IAAPerspective(p=1)
    augmented = aug(**param)
    image, mask = augmented['image'], augmented['mask']
    cv2.imwrite(os.path.join(saveto,'aug_perspective_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_perspective_mask.png'),mask)


def aug_noise(param,saveto):
    aug = Compose([Downscale(scale_min=0.25, scale_max=0.5, p=1),
            GaussNoise(p=1)])
    augmented = aug(**param)
    image, mask = augmented['image'], augmented['mask']
    cv2.imwrite(os.path.join(saveto,'aug_noise_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_noise_mask.png'),mask)


def aug_crop(param,saveto):
    original_height, original_width = param['mask'].shape
    aug = RandomSizedCrop(p=1, 
                      min_max_height=(100,300),
                      w2h_ratio=3, 
                      height=original_height, 
                      width=original_width)
    augmented = aug(**param)
    image, mask = augmented['image'], augmented['mask']
    cv2.imwrite(os.path.join(saveto,'aug_crop_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_crop_mask.png'),mask)


def aug_brightness(param,saveto):
    aug = OneOf([RandomBrightness(limit=0.5, p=1), RandomContrast(p=1)])
    augmented = aug(**param)
    image, mask = augmented['image'], augmented['mask']
    cv2.imwrite(os.path.join(saveto,'aug_brightness_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_brightness_mask.png'),mask)


def aug_lane_erase(param,saveto):
    image, mask = param['image'], param['mask']
    original_height, original_width = mask.shape
    X, Y = np.nonzero(mask)
    index = [(X[i],Y[i]) for i in range(len(X)) \
        if X[i] in range(200,original_height-50) and \
            Y[i] in range(100,original_width-50)]
    x,y = index[np.random.randint(1,len(index))]
    color = [0,255]
    color = color[np.random.randint(0,2)]
    for ch in range(0,3):
        image[x-25:x+25, y-25:y+25, ch] = color
    mask[x-25:x+25, y-25:y+25] = color
    cv2.imwrite(os.path.join(saveto,'aug_erase_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_erase_mask.png'),mask) 


def aug_compose(param,saveto):
    aug = Compose([
        ShiftScaleRotate(shift_limit=0.3, 
                        scale_limit=0.3, 
                        rotate_limit=30, 
                        border_mode=0),
        HorizontalFlip(),
        IAAPerspective(),
        OneOf([Downscale(scale_min=0.25, scale_max=0.5), GaussNoise()]),
        OneOf([RandomBrightness(limit=0.5), RandomContrast()])
    ])
    augmented = aug(**param)
    image, mask = augmented['image'], augmented['mask']
    if (np.random.randint(0,2)):
        original_height, original_width = mask.shape
        X, Y = np.nonzero(mask)
        index = [(X[i],Y[i]) for i in range(len(X)) \
            if X[i] in range(200,original_height-50) and \
                Y[i] in range(100,original_width-50)]
        x,y = index[np.random.randint(1,len(index))]
        color = [0,255]
        color = color[np.random.randint(0,2)]
        for ch in range(0,3):
            image[x-25:x+25, y-25:y+25, ch] = color
        mask[x-25:x+25, y-25:y+25] = color
    cv2.imwrite(os.path.join(saveto,'aug_compose_image.png'),image)
    cv2.imwrite(os.path.join(saveto,'aug_compose_mask.png'),mask)


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

        data = {'image': img_data, 
                'image': pc_data, 
                'mask': pc2_data, 
                'mask': lane_data, 
                'mask': road_data}
        aug_rotate(param=data,saveto=saveto)
        aug_rotate(param=data,saveto=saveto)
        aug_flip(param=data,saveto=saveto)
        aug_perspective(param=data,saveto=saveto)
        aug_noise(param=data,saveto=saveto)
        aug_crop(param=data,saveto=saveto) 
        aug_brightness(param=data,saveto=saveto)
        aug_lane_erase(param=data, saveto=saveto)
        aug_compose(param=data, saveto=saveto)
        aug_compose(param=data, saveto=saveto)
        aug_compose(param=data, saveto=saveto)



if __name__ == "__main__":
    img_path = '../data/img/um_000000.png'
    img_path2 = '../data/img/um_000006.png'
    label_path = '../data/label/lane/1.png'
    saveto = '../result/aug'
    img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)

    data = {'image': img, 'mask': mask}
    aug_rotate(param=data,saveto=saveto)
    # aug_flip(param=data,saveto=saveto)
    # aug_perspective(param=data,saveto=saveto)
    # aug_noise(param=data,saveto=saveto)
    # aug_crop(param=data,saveto=saveto)
    # aug_brightness(param=data,saveto=saveto)
    # aug_lane_erase(param=data, saveto=saveto)
    # aug_compose(param=data, saveto=saveto)