import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torch.utils.data as data
import cv2


CATEGORY={'bike-packing':2, 'bmx-bumps':2, 'bmx-trees':2, 'boxing-fisheye':1,
                'breakdance':1, 'breakdance-flare':1, 'cat-girl':1, 'color-run':1,
                'crossing':1, 'dance-jump':1, 'dance-twirl':1, 'dancing':1, 
                'disc-jockey':1, 'hike':1, 'hockey':1, 'india':1, 'judo':1,
                'kid-football':1, 'lady-running':1, 'loading':1, 'lucia':1}

RATIO_THRESH = 0.0005


class DavisDataset(data.Dataset):
    def __init__(self, datasetDir='.', perfix='480p', transforms=None):
        super(DavisDataset, self).__init__()
        self.datasetDir = datasetDir
        self.perfix = perfix
        self.imageDir = self._get_dataDir()
        self.annDir = self._get_dataDir(ann=True)
        self.transforms = transforms
        self.videos = self._get_videos()                 # a list of pathes of video directory
        self.frame_range = 5
        self.ratio_thresh = RATIO_THRESH

    def _get_dataDir(self, ann=False):
        if ann:
            data_type = 'Annotations'
        else:
            data_type = 'JPEGImages'

        return os.path.join(self.datasetDir, 'DAVIS', data_type, self.perfix)

    def _get_videos(self):
        videos = []
        for cat in CATEGORY:
            # video_path = os.path.join(self.imageDir, cat)
            # video_ann_path = os.path.join(self.annDir, cat)
            # videos["frame"].append([video_path, CATEGORY[cat]])
            # videos["ann"].append([video_ann_path, CATEGORY[cat]])
            videos.append([cat, CATEGORY[cat]])
        return videos

    def _get_bbox_center_from_mask(self, mask):
        '''all coordinates are in opencv axises
        '''
        mask_idx = mask.nonzero()

        x1 = np.min(mask_idx[1])
        y1 = np.min(mask_idx[0])
        x2 = np.max(mask_idx[1])
        y2 = np.max(mask_idx[0])

        cx = np.mean(mask_idx[1])
        cy = np.mean(mask_idx[0])

        bbox = self._xyxy2xywh([x1, y1, x2, y2])
        center = np.array([cx,cy])

        return bbox, center


    def get_positive_pair(self, index):

        mask_code = self.videos[index][1]
        template, search = self.get_pair(index)

        template_mask = (np.array(template[1]) == mask_code).astype(np.uint8)
        search_mask = (np.array(search[1]) == mask_code).astype(np.uint8)

        th, tw = template_mask.shape
        sh, sw = search_mask.shape
        ################################################
        # need to be fix 
        # 这里当template里面的目标太小, 我直接将其替换为第0帧的template,并且让search等于template
        # 这就要求每个视频的第0帧都是有效的. 当然这并不一定，因此该处理有待优化
        ###############################################
        if template_mask.sum() < th * tw * self.ratio_thresh:
            template, _ = self.get_pair(index, frame_no=0)
            search = template.copy()
        elif search_mask.sum() < sh * sw * self.ratio_thresh:
            search = template.copy()

        return template, search

    def get_pair(self, index, frame_no=None):
        video_name = self.videos[index][0]
        video_path = os.path.join(self.imageDir, video_name)
        video_ann_path = os.path.join(self.annDir, video_name)

        frames = os.listdir(video_path)
        frames.sort()

        if frame_no is not None:
            template_frame = frame_no
        else:
            template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)

        return self.get_image_anno(video_name, template_frame), \
            self.get_image_anno(video_name, search_frame)

    def get_image_anno(self, video_name, frame):
        image_path = os.path.join(self.imageDir, video_name, frame)
        ann_path = os.path.join(self.annDir, video_name, frame.replace('jpg', 'png'))
        image = Image.open(image_path)
        image_anno = Image.open(ann_path)
        return [image, image_anno, image_path]

    def _xyxy2xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        return np.array([x1, y1, w, h])

    def get_36_coordinates(self, c_x, c_y, pos_mask_contour):
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

    def get_valid_center_from_fm(self, fm_size, c_x, c_y):
        valid_centers = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if c_x + i in range(0, fm_size) and c_y + j in range(0, fm_size):
                    valid_centers.append(np.array((c_x + i, c_y + j)))
        return valid_centers

    def gen_gt_class(self, fm_size, valid_centers):
        gt_class = np.zeros((fm_size, fm_size))
        for c in valid_centers:
            gt_class[int(c[0]), int(c[1])] = 1
        return torch.from_numpy(gt_class).to(torch.long)

    def coord_transform(self, coords, mode='f2o'):
        new_coords = []
        if mode == 'f2o':
            for c in coords:
                new_coords.append(8 * c + 31)
        elif mode == 'o2f':
            for c in coords:
                new_coords.append(round((c - 31) / 8))
        return new_coords

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        mask_code = self.videos[index][1]

        # get image annotation
        template, detection = self.get_positive_pair(index)

        # get mask annotation
        template_mask = (np.array(template[1]) == mask_code).astype(np.uint8)
        detection_mask = (np.array(detection[1]) == mask_code).astype(np.uint8)

        # get bounding box from anno
        tbbox, tc = self._get_bbox_center_from_mask(template_mask)
        sbbox, sc = self._get_bbox_center_from_mask(detection_mask)

        ################################################################
        # TODO 
        # 需要将template, detection, tbbox, sbbox, sc转换到各自的坐标系下，
        # 并且需要将detection_mask变换到255坐标系下
        # template和detection 是一个list, 里面的元素分别为: 原始PIL图片， PIL格式的mask, 图片路径
        ################################################################

        # find center in feature map
        f_cx = round(max(0, (sc[0] - 31) / 8))
        f_cy = round(max(0, (sc[1] - 31) / 8))

        # operation on feature map coordinate
        valid_centers = self.get_valid_center_from_fm(25, f_cx, f_cy)
        gt_class = self.gen_gt_class(25, valid_centers)

        valid_centers_in_ori = self.coord_transform(valid_centers, 'f2o')

        contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours)  # size: n * 1 * 2
        contours = contours.reshape(-1, 2)  # 此处为opencv的x, y坐标，后处理需要交换

        distances = torch.ones(25, 25, 36) * 1e-6
        new_coordinates = []
        for i, c in enumerate(valid_centers_in_ori):
            dst, new_coord = self.get_36_coordinates(c[0], c[1],
                                                     contours)  # distance is a list, new_coord is a dictionary which keys are angles(0 ~ 360, 10)
            distances[int(valid_centers[i][0]), int(valid_centers[i][1]), :] = dst
            new_coordinates.append(new_coord)

        # distances = torch.cat(distances[:], 0)
        distance, new_coord = self.get_36_coordinates(sc[0], sc[1], contours)

        if self.transforms is not None:
            template = self.transforms(template)
            detection = self.transforms(detection)

        meta = {}
        meta['imgId'] = {
            'template': template[-1],
            'detection': detection[-1],
        }
        meta['template'] = np.array(template[0])  # 模板 3 * 127 * 127
        meta['detection'] = np.array(detection[0])  # 检测 3 * 256 * 256
        meta['bbox_of_detection'] = sbbox  # 检测帧坐标系下的bbox,
        meta['center'] = sc  # 检测帧坐标系下中心坐标
        meta['distance'] = distance  # 检测帧坐标系下距离
        meta['coords'] = new_coord  # 字典， 键为角度，键值为距离
        meta['targets'] = {
            'distances': distances,  # 25 * 25 * 36
            'gt_class': gt_class,  # 25 * 25
        }
        # meta['valid_centers'] = valid_centers
        meta['mask'] = detection_mask

        return meta




if __name__ == "__main__":
    # root_dir = './DAVIS/Annotations/480p/'
    # for cat in CATEGORY:

    #     fileDir = os.path.join(root_dir, cat)
    #     fileNames = os.listdir(fileDir)
    #     fileNames.sort()
    #     print(fileNames)

    #     for file in fileNames:
    #         filepath = os.path.join(root_dir, cat, file)
    #         img = Image.open(filepath)

    #         img_np = np.array(img)
    #         mask = (img_np == CATEGORY[cat]).astype(np.uint8) # np.array(img_np == CATEGORY[cat], dtype=np.uint8)
    #         if mask.sum() < mask.shape[0] * mask.shape[1] * RATIO_THRESH:
    #             continue
    #         # print(mask.sum())
    #         # print(mask.shape)

    #         mask_cv = np.expand_dims(mask, axis=2)
    #         mask_cv = np.concatenate([mask_cv, mask_cv, mask_cv], axis=2)
    #         mask_cv *= 255

    #         cv2.imshow("mask", mask_cv)

    #         cv2.waitKey(25)

    dataset = DavisDataset()
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    def get_contours_from_polar(cx, cy, polar_coords):
        new_coords = []
        for angle, dst in polar_coords.items():
            x = cx + dst.double() * np.cos(angle * np.pi / 180)
            y = cy + dst.double() * np.sin(angle * np.pi / 180)
            new_coords.append([x, y])
        return new_coords


    for i, Data in enumerate(loader):
        temp = (Data)
        print(i)
        if i == 100:
            break

        image_id = temp['imgId']
        print(image_id['template'], '\n', image_id['detection'])
        template = temp['template'][0].numpy()
        print(template.shape)
        detection = temp['detection'][0].numpy()
        print(type(detection))
        center = temp['center'][0].numpy()
        distance = temp['distance'][0].numpy()
        coords = temp['coords']
        # print(coords)
        # GT_reg = temp['targets']['distances'][0].reshape(-1, 36)
        # GT_cls = temp['targets']['gt_class'][0]
        bbox = temp['bbox_of_detection'][0]
        GT_mask = temp['mask'][0].numpy() * 255
        
        # print(GT_cls.shape)
        # pos_inds = torch.nonzero(GT_cls.view(-1) > 0).squeeze(1)
        # print(GT_reg[pos_inds[0]])
        # print(GT_reg[pos_inds[1]])
        # print(GT_reg[pos_inds[2]])
        #######################################
        # 中心以及distance可视化

        #plt.imshow(detection.permute(1, 2, 0))
        #plt.imshow(GT_mask)

        new_coords = get_contours_from_polar(center[0], center[1], coords)

        mask = np.array([GT_mask, GT_mask, GT_mask]).transpose(1,2,0)
        
        for i in range(len(new_coords)):
            cv2.line(detection, (int(center[0]), int(center[1])), (int(new_coords[i][0]), int(new_coords[i][1])), (0,0,255), 2)

        imgs = np.hstack([template, detection, mask])
        cv2.imshow("imgs", imgs)
        cv2.waitKey(0)