# import sys
# sys.path.append("..")
import cv2
import os
import numpy as np

# evaluate the accuracy between pred and gt, using F1-score
def f1(pred, gt):
    pred = cv2.imread(pred,cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt,cv2.IMREAD_GRAYSCALE)
    # resize
    h,w = gt.shape
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    # normalization
    np.putmask(gt, gt>0, 1.0)
    np.putmask(pred, pred>=0.5, 1.0)
    np.putmask(pred, pred<0.5, 0)
    # indices
    total = h*w
    result = gt*2 + pred
    tn,fp,fn,tp = np.bincount(result)
    # outliers
    if tp+fp==0:
        if fn==0:
            precise = 1
        else:
            precise = 0.5
    else:
        precise = np.divide(tp,tp+fp)

    if tp+fn==0:
        if fp==0:
            recall = 1
        else:
            recall = 0.5
        balance_acc = tn / total
    else:
        recall = np.divide(tp,tp+fn)
        balance_acc = 2*tp*tn + fn*tn + fp*tp

    F1 = np.divide(2*precise*recall,precise+recall)

    return precise, recall, F1, balance_acc


def f1_batch(pred_path, gt_path):
    path_lst = (pred_path, gt_path)
    PATH = []
    for path in path_lst: #rank files
        files = os.listdir(path)
        files.sort()
        PATH.append(files)

    precise = []
    recall = []
    F1 = []
    Bacc = []
    for pred,gt in zip(PATH[0],PATH[1]):
        p, r, f, b = f1(pred,gt)
        precise.append(p)
        recall.append(r)
        F1.append(f)
        Bacc.append(b)

    return np.mean(precise), np.mean(recall), np.mean(F1), np.mean(Bacc)


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
    #create_list('data/pred/')
    p,r,f,b = f1_batch()
    print('precise: %.3f\nrecall: %.3f\nf1-score: %.3f\nb-acc: %.3f' % (p, r, f, b))
    #overaly_batch('list.txt')
    # overlay('data/lane_binary/orig_test_lane_b_um_000029.png','data/train/orig_test_um_000029.png','data/overlay/29.png')
    # add_img('/media/txxzzh/B2D21A09D219D309/Automobile_THU/dataset/kitti-lanes/data_road/testing/image_2/um_000029.png','data/overlay/29.png')
    #overaly_batch_bl(path_o='data/baseline/label/')
