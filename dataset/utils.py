from torch.utils.data import Dataset
import os
import cv2
import random
import math
import numpy as np

def augment_strokes(strokes, prob=0.0):
    """ Perform data augmentation by randomly dropping out strokes """
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = list(candidate)
            prev_stroke = list(stroke)
            result.append(stroke)
    return np.array(result)


def seq_3d_to_5d(stroke, max_len=250):
    """ Convert from 3D format (npz file) to 5D (sketch-rnn paper) """
    result = np.zeros((max_len, 5), dtype=np.float32)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result

def rescale(X, ratio=0.85):
    """ Rescale the image to a smaller size """
    h, w = X.shape

    h2 = int(h*ratio)
    w2 = int(w*ratio)

    X2 = cv2.resize(X, (w2, h2), interpolation=cv2.INTER_AREA)

    dh = int((h - h2) / 2)
    dw = int((w - w2) / 2)

    res = np.copy(X)
    res[:,:] = 255
    res[dh:(dh+h2),dw:(dw+w2)] = X2

    return res


def rotate(X, angle=15):
    """ Rotate the image """
    h, w = X.shape
    rad = np.deg2rad(angle)

    nw = ((abs(np.sin(rad)*h)) + (abs(np.cos(rad)*w)))
    nh = ((abs(np.cos(rad)*h)) + (abs(np.sin(rad)*w)))

    rot_mat = cv2.getRotationMatrix2D((nw/2,nh/2),angle,1)
    rot_move = np.dot(rot_mat,np.array([(nw-w)/2,(nh-h)/2,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]

    res_w = int(math.ceil(nw))
    res_h = int(math.ceil(nh))

    res = cv2.warpAffine(X,rot_mat,(res_w,res_h),flags=cv2.INTER_LANCZOS4, borderValue=255)
    res = cv2.resize(res,(w,h), interpolation=cv2.INTER_AREA)

    return res


def translate(X, dx=5,dy=5):
    """ Translate the image """
    h, w = X.shape
    M = np.float32([[1,0,dx],[0,1,dy]])
    res = cv2.warpAffine(X,M,(w,h), borderValue=255)

    return res