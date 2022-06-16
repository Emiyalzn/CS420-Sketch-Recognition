from torch.utils.data import Dataset
import os
import cv2
import random
import math
import numpy as np
from copy import deepcopy

def bbox(points):
    return np.amin(points, axis=0), np.amax(points, axis=0)

def rotate_mat(degree):
    m = np.identity(3, 'float32')
    theta_rad = degree * np.pi / 180.0
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    m[0, 0] = cos_theta
    m[0, 1] = sin_theta
    m[1, 0] = -sin_theta
    m[1, 1] = cos_theta

    return m

def scale_mat(sx, sy=None):
    if sy is None:
        sy = sx

    m = np.identity(3, 'float32')
    m[0, 0] = sx
    m[1, 1] = sy
    return m

def translate_mat(delta_x, delta_y):
    m = np.identity(3, 'float32')
    m[0, 2] = delta_x
    m[1, 2] = delta_y
    return m

def random_affine_transform(points, scale_factor=0.2, rot_thresh=30.0):
    bbox_min, bbox_max = bbox(points)
    bbox_center = (bbox_min + bbox_max) / 2.0
    x_scale_factor = 1.0 - np.random.random() * scale_factor
    y_scale_factor = 1.0 - np.random.random() * scale_factor
    rot_degree = (np.random.random() - 0.5) * 2 * rot_thresh

    t_0 = translate_mat(-bbox_center[0], -bbox_center[1])
    s_1 = scale_mat(x_scale_factor, y_scale_factor)
    r_2 = rotate_mat(rot_degree)
    t_3 = translate_mat(bbox_center[0], bbox_center[1])
    transform_ = np.matmul(t_3, np.matmul(r_2, np.matmul(s_1, t_0)))

    transformed_points = transform(points, transform_)
    return transformed_points

def transform(points, mat):
    temp_pts = np.ones(shape=(len(points), 3), dtype='float32')
    temp_pts[:, 0:2] = np.array(points, dtype='float32')

    transformed_pts = np.matmul(temp_pts, mat.T)
    return transformed_pts[:, 0:2]

def random_remove_strokes(strokes, prob=0.2):
    result = []
    for i in range(len(strokes)):
        stroke = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if stroke[2] == 0:
            urnd = np.random.rand()
            if urnd < prob:
                stroke[2] = 1
        result.append(stroke)
    return np.array(result)

def seqlen_remove_strokes(strokes, drop_prob=0.2):
    alpha = 2
    beta = 0.5
    length = len(strokes)
    count = 0
    probs = []
    for i in range(length):
        if strokes[i][2] == 1:
            probs.append(0.)
            continue
        count += 1
        prob = np.exp(alpha * count) / np.exp(beta * np.sqrt(strokes[i][0] ** 2 + strokes[i][1] ** 2))
        probs.append(prob)
    probs = np.array(probs) / np.sum(probs)
    drop_indices = np.random.choice(np.arange(length), int(drop_prob * count), False, p=probs)

    result = deepcopy(strokes)
    for ind in drop_indices:
        result[ind][2] = 1
    return result

def seqlen_remove_points(points, drop_prob=0.2):
    alpha = 2
    beta = 0.5
    length = len(points)
    count = 0
    probs = [0.]
    for i in range(1, length):
        if points[i][2] == 1:
            probs.append(0.)
            continue
        count += 1
        prob = np.exp(alpha * count) / np.exp(beta * np.sqrt((points[i][0]-points[i-1][0]) ** 2 + (points[i][1]-points[i-1][1]) ** 2))
        probs.append(prob)
    probs = np.array(probs) / np.sum(probs)
    drop_indices = np.random.choice(np.arange(length), int(drop_prob * count), False, p=probs)

    result = deepcopy(points)
    for ind in drop_indices:
        result[ind][2] = 1
    return result

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

def seq_5d_to_3d(stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(stroke)):
        if stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = stroke[0:l, 0:2]
    result[:, 2] = stroke[0:l, 3]
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