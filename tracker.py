# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch.nn.functional as F
import cv2
import torch
from torchvision import transforms

from config import cfg


class MyTracker(object):
    def __init__(self, model, cfg):
        super(MyTracker, self).__init__()
        self.cfg = cfg
        self.score_size = cfg.SCORE_SIZE
        hanning = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning)
        self.score_size_up = cfg.UPSIZE
        self.model = model
        self.model.eval()
        self.mean = [0.471, 0.448, 0.408]
        self.std =[0.234, 0.239, 0.242]
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])



    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        cv2.imshow("1", im_patch)
        cv2.waitKey()
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = self.trans(im_patch.squeeze()).unsqueeze(0)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch

    def _convert_score(self, score):
        score = F.softmax(score[:, :, :, :], dim=1).data[:, 1, :, :].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, mask, boundary):
        print(boundary[1],boundary[0])
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        max_dis = np.sqrt((width / 2)**2 + (height / 2)**2)
        for i in range(len(mask)):
            mask[i] = max(0, min(mask[i], max_dis))
        return cx, cy, width, height, mask

    def init(self, img, bbox, mask):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        self.mask = np.array(mask)
        #print("bbox中心",self.center_pos[0], self.center_pos[1])
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop

        z_crop = self.crop_template(img)

        self.model.template(z_crop)

    def crop_template(self, img):
        # calculate z crop size
        w_z, h_z = self.size + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        z = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        return z

    def crop_search_region(self, img):
        w_z, h_z = self.size + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)

        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        return x

    # get bbox
    def get_bbox(self, p_coor, cen, bboxes):
        max_r, max_c = p_coor[0], p_coor[1]
        bbox = bboxes[:, max_r, max_c]
        l, t, r, b = bbox
        cen_value = cen[max_r, max_c]
        return np.array([l, t, r, b]), cen_value

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, bboxes, size_ori, penalty_lk):
        bboxes_w = bboxes[0, :, :] + bboxes[2, :, :]
        bboxes_h = bboxes[1, :, :] + bboxes[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(size_ori[0]*self.scale_z, size_ori[1]*self.scale_z))
        r_c = self.change((size_ori[0] / size_ori[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty


    def get_size_from_mask(self, cx, cy, distances,size):
        x = []
        y = []
        angle = 0
        for dst in distances:

            x.append(dst * np.cos(angle * np.pi / 180))
            y.append(dst * np.sin(angle * np.pi / 180))
            angle += 10
        w = min(max(x) - min(x),size[0])
        h = min(max(y) - min(y),size[1])
        bbox_cen_x = cx# + max(min(x), -w/2) + w / 2
        bbox_cen_y = cy #+ max(min(y), -h/2) + h / 2

        return int(w), int(h), int(bbox_cen_x), int(bbox_cen_y)

    def track(self, img, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        x_crop = self.crop_search_region(img)
        score, masks, cen = self.model.track(x_crop)

        cen = torch.sigmoid(cen)

        score = self._convert_score(score)
        score = score.squeeze()

        max_r_1, max_c_1 = np.unravel_index(score.argmax(), score.shape)
        cen = cen.data.cpu().numpy()
        cen = cen.squeeze()

        masks = masks.squeeze().permute(1,2,0)
        masks = masks.data.cpu().numpy()


        if self.cfg.hanming:
            h_score = score*(1-hp['w_lr']) + self.window * hp['w_lr']
        else:
            h_score = score

        score_up = cv2.resize(h_score, (193, 193), interpolation=cv2.INTER_CUBIC)
        cen_up = cv2.resize(cen, (193, 193), interpolation=cv2.INTER_CUBIC)
        scale_resmap = 193 / 25
        res_map_up = score_up #* cen_up


        max_r_up, max_c_up = np.unravel_index(res_map_up.argmax(), res_map_up.shape)

        max_r, max_c = int(round(max_r_up/scale_resmap)), int(round(max_c_up/scale_resmap))

        max_r_up += 31
        max_c_up += 31
        # print("255最大", max_r_up, max_c_up)
        mask = masks[max_r, max_c] / self.scale_z


        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([255, 255]) - 1.) / 2.
        disp_ori = disp #/ self.scale_z
        #print("偏差",disp,disp_ori)
        ave_cx = disp_ori[1] + self.center_pos[0]
        ave_cy = disp_ori[0] + self.center_pos[1]

        #print("原图重心",ave_cx,ave_cy)

        ave_w, ave_h, bbox_cen_x, bbox_cen_y = self.get_size_from_mask(ave_cx, ave_cy, mask,self.size)
        #print("原图中心切割前", bbox_cen_x, bbox_cen_y)
        #print("长宽",ave_w,ave_h)
        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['pk'])
        lr = penalty * score[max_r, max_c] * hp['lr']
        #print("lr",lr)
        width = lr * ave_w + (1-lr) * self.size[0]
        height = lr * ave_h + (1-lr) * self.size[1]
        mask = lr * mask + (1-lr)*self.mask

        # clip boundary
        cx, cy, width, height, mask = self._bbox_clip(bbox_cen_x, bbox_cen_y, width, height, mask, img.shape[:2])
        #print("原图中心切割后",cx,cy)
        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        self.mask = mask

        return {
                "bbox": [self.center_pos[0], self.center_pos[1], self.size[0], self.size[1]],
                "mask": mask,
                "mask_cen": [ave_cx, ave_cy]
               }