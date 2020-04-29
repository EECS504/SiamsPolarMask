
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from PIL import Image, ImageOps, ImageStat,ImageDraw
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob
import numpy as np
from config import cfg
from model import Mymodel
from tracker import MyTracker

torch.set_num_threads(1)
def get_contours_from_polar(cx, cy, polar_coords):
    new_coords = []
    angle = 0
    for dst in polar_coords:
        x = cx + dst * np.cos(angle * np.pi / 180)
        y = cy + dst * np.sin(angle * np.pi / 180)
        new_coords.append([x, y])
        angle += 10
    return new_coords

def get_36_coordinates(c_x, c_y, pos_mask_contour):
    # 输入为opencv坐标系下的中心点x,y以及contuor
    ct = pos_mask_contour
    x = torch.Tensor(ct[:, 0] - c_x)  # opencv x, y交换
    y = torch.Tensor(ct[:, 1] - c_y)

    # torch.atan2的输入非常迷惑， 第一个参数必须是y坐标，第二个参数是x
    angle = torch.atan2(y, x) * 180 / np.pi
    angle[angle < 0] += 360
    angle = angle.int()
    # dist = np.sqrt(x ** 2 + y ** 2)
    dist = torch.sqrt(x ** 2 + y ** 2)
    angle, idx = torch.sort(angle)
    dist = dist[idx]
    # ct2 = ct[idx]

    # 生成36个角度
    new_coordinate = {}
    for i in range(0, 360, 10):
        if i in angle:
            d = dist[angle == i].max()
            new_coordinate[i] = d
        elif i + 1 in angle:
            d = dist[angle == i + 1].max()
            new_coordinate[i] = d
        elif i - 1 in angle:
            d = dist[angle == i - 1].max()
            new_coordinate[i] = d
        elif i + 2 in angle:
            d = dist[angle == i + 2].max()
            new_coordinate[i] = d
        elif i - 2 in angle:
            d = dist[angle == i - 2].max()
            new_coordinate[i] = d
        elif i + 3 in angle:
            d = dist[angle == i + 3].max()
            new_coordinate[i] = d
        elif i - 3 in angle:
            d = dist[angle == i - 3].max()
            new_coordinate[i] = d

    distances = torch.zeros(36)

    for a in range(0, 360, 10):
        if not a in new_coordinate.keys():
            new_coordinate[a] = torch.tensor(1e-6)
            distances[a // 10] = 1e-6
        else:
            distances[a // 10] = new_coordinate[a]

    return distances, new_coordinate

def _xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    return np.array([x1, y1, w, h])

def _get_bbox_center_from_mask(mask):
    '''all coordinates are in opencv axises
    '''
    mask_idx = mask.nonzero()

    x1 = np.min(mask_idx[1])
    y1 = np.min(mask_idx[0])
    x2 = np.max(mask_idx[1])
    y2 = np.max(mask_idx[0])

    cx = np.mean(mask_idx[1])
    cy = np.mean(mask_idx[0])

    bbox = _xyxy2xywh([x1, y1, x2, y2])
    center = np.array([cx, cy])

    return center, bbox

def get_frames(video_name):
    images = glob(os.path.join(video_name, '000*.jpg'))
    imagess = sorted(images, key=lambda x: int(x[-9:-4]))
    for img in imagess:
        frame = cv2.imread(img)
        yield frame

def get_bbox_mask(ann_path):
    ann = np.array(Image.open(ann_path + "/00000.png"))
    center, bbox = _get_bbox_center_from_mask(ann)
    contours_template, _ = cv2.findContours(ann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_template = np.concatenate(contours_template).reshape(-1, 2)
    mask, _ = get_36_coordinates(center[0], center[1], contours_template)
    # print("bbox重心", center[0],center[1])
    return bbox, mask, center



def main():

    # load config
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = Mymodel()

    # load model
    checkpoint_path = './checkpoint/best_checkpoint.pytorch'
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # build tracker
    tracker = MyTracker(model, cfg.TRACK)

    # hp_search

    hp = {'lr': 0.3, 'pk': 0.04, 'w_lr': 0.7}

    image_path = "./DAVIS/JPEGImages/480p/hike/"
    ann_path = "./DAVIS/Annotations/480p/hike/"

    first_frame = True
    video_name = "test"
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(image_path):

        if first_frame:
            bbox, mask ,center= get_bbox_mask(ann_path)
            coords1 = get_contours_from_polar(center[0], center[1], mask)
            for i in range(len(coords1)):
                cv2.line(frame, (int(center[0]), int(center[1])), (int(coords1[i][0]), int(coords1[i][1])),(0, 255, 0), 1)
            tracker.init(frame, bbox, mask)
            first_frame = False
            cv2.rectangle(frame, (bbox[0], bbox[1] ),
                          (bbox[0] + bbox[2] , bbox[1] + bbox[3]),
                          (0, 255, 0), 3)
        else:
            outputs = tracker.track(frame, hp)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0] - bbox[2] // 2, bbox[1] - bbox[3] // 2 - 40),
                          (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2 - 40),
                          (0, 255, 0), 3)
            coords = get_contours_from_polar(outputs['mask_cen'][0], outputs['mask_cen'][1], outputs['mask'])
            for i in range(len(coords)):
                cv2.line(frame, (int(outputs['mask_cen'][0]), int(outputs['mask_cen'][1] - 40)), (int(coords[i][0]), int(coords[i][1] - 40)),(0, 255, 0), 3)

        cv2.imshow(video_name, frame)
        cv2.waitKey()


if __name__ == '__main__':
    main()
