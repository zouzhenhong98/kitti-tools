# import sys
# sys.path.append("..")
import cv2
import os
import numpy as np

img_pred = '/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/SCNN-kitti'
img_gt = '../data/label/lane/5.png'

# evaluate the accuracy between pred and gt, using F1-score
def f1(pred, gt):
    pred = cv2.imread(pred,cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt,cv2.IMREAD_GRAYSCALE)
    # resize
    h,w = gt.shape[:2]
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    # normalization
    #gt = gt/255
    pred = pred/255
    np.putmask(gt, gt>0, 1.0)
    np.putmask(pred, pred>0, 1.0)
    # np.putmask(pred, pred, 0.0)
    # # indices
    total = h*w
    gt_lane = np.count_nonzero(gt)
    pred_lane = np.count_nonzero(pred)
    correct_lane = np.count_nonzero(np.multiply(gt,pred))
    
    precise = np.divide(correct_lane,pred_lane)
    recall = np.divide(correct_lane,gt_lane)
    if recall==0:
        F1 = 0
    else:
        F1 = np.divide(2*precise*recall,precise+recall)

    return precise, recall, F1, correct_lane


def f1_batch(lst):
    PRED = []
    GT = []
    precise = []
    recall = []
    F1 = []
    cl = []
    with open(lst) as f:
        content = f.readlines()
        for c in content:
            if c[:9]=='orig_test':
                PRED.append('data/pred/' + c[:-1])
                GT.append('data/lane_binary/' + c[:10]+'lane_b_'+c[10:-1])
            else:
                PRED.append('data/pred/' + c[:-1])
                GT.append('data/lane_binary/' + c[:11]+'lane_b_'+c[11:-1])

    for i in range(len(PRED)):    
        p, r, f, CL= f1(PRED[i],GT[i])
        precise.append(p)
        recall.append(r)
        F1.append(f)
        cl.append(CL)

    return np.mean(precise), np.mean(recall), np.mean(F1)


# create a index txt for files under the path
def create_list(path_to_create):
    name = []
    for a,b,files in os.walk(path_to_create):
        for f in files:
            name.append(f+'\n')
    with open('list.txt', 'w+') as f:
        f.writelines(name)


def overlay(gt_o, pred_o, saveto):
    # print(gt_o)
    # print(type(gt_o))
    # print(os.listdir())
    # print(os.path.exists(str(gt_o)))
    gt_o = cv2.imread(gt_o, cv2.IMREAD_GRAYSCALE)
    pred_o = cv2.imread(pred_o, cv2.IMREAD_GRAYSCALE)
    # # indicated resize
    # h,w = (300,1000)
    # pred_o = cv2.resize(pred_o, (w, h), interpolation=cv2.INTER_AREA)
    # gt_o = cv2.resize(gt_o, (w, h), interpolation=cv2.INTER_AREA)
    # dowbsampling
    # h,w = pred_o.shape[:2]
    # gt_o = cv2.resize(gt_o, (w, h), interpolation=cv2.INTER_AREA)
    # upsampling
    h,w = gt_o.shape[:2]
    pred_o = cv2.resize(pred_o, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_o = pred_o * 0.8 + gt_o * 125
    cv2.imwrite(saveto, pred_o)


def overaly_batch(lst_o):
    with open(lst_o) as f:
        for line in f.readlines():
            if line[:10]=='orig_test_':
                gt_o = './data/lane_binary/'+line[:10]+'lane_b_'+line[10:]
                pred_o = './data/pred/'+line
            else:
                gt_o = './data/lane_binary/'+line[:11]+'lane_b_'+line[11:]
                pred_o = './data/pred/'+line
            
            overlay(gt_o[:-1], pred_o[:-1], saveto='./data/overlay/'+line)

def overaly_batch_bl(path_o):
    for a,b,c in os.walk(path_o):
        for f in c:
            overlay(gt_o='data/baseline/label/'+f, pred_o='data/baseline/test_completion_segmentation_segmentation/'+f, saveto='./data/baseline/'+f)


def add_img(raw, label):
    raw = cv2.imread(raw)
    label = cv2.imread(label)
    com = cv2.addWeighted(raw,0.8,label,0.5,0)
    cv2.imshow('1',com)
    cv2.waitKey(0)


if __name__ == "__main__":
    create_list('data/pred/')
    p,r,f = f1_batch('list.txt')
    print('precise: %.3f\nrecall: %.3f\nf1-score: %.3f' % (p, r, f))
    overaly_batch('list.txt')
    # overlay('data/lane_binary/orig_test_lane_b_um_000029.png','data/train/orig_test_um_000029.png','data/overlay/29.png')
    # add_img('/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road/testing/image_2/um_000029.png','data/overlay/29.png')
    overaly_batch_bl(path_o='data/baseline/label/')
