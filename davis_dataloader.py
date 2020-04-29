import numpy as np
from PIL import Image, ImageOps, ImageStat,ImageDraw
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

    def transform_one_point_cords(self, original, bbox, cords_of_one_point):
        bbox_xywh =  np.array([bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]],np.float32)  # 将标注ann中的bbox从左上点的坐标+bbox长宽 -> 中心点的坐标+bbox的长宽

        original_image_w, original_image_h = original.size
        cx, cy, tw, th = bbox_xywh

        p = round((tw + th) / 2, 2)
        template_square_size = int(np.sqrt((tw + p) * (th + p)))  # a
        detection_square_size = int(template_square_size * 2)  # A = 2a

        detection_lt_x, detection_lt_y = cx - detection_square_size // 2, cy - detection_square_size // 2
        detection_rb_x, detection_rb_y = cx + detection_square_size // 2, cy + detection_square_size // 2

        x, y = cords_of_one_point[0], cords_of_one_point[1]

        x_in_detection_before_resize, y_in_detection_before_resize = x - detection_lt_x, y - detection_lt_y
        x_in_detection_after_resize,  y_in_detection_before_resize = (x_in_detection_before_resize/detection_square_size*255), (y_in_detection_before_resize/detection_square_size*255)

        return (x_in_detection_after_resize,  y_in_detection_before_resize)

    def transform_cords(self, image, bbox, segmentation_in_original_image):
        mean_template_and_detection = tuple(map(round, ImageStat.Stat(image).mean))
        bbox_xywh = np.array([bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]],
                             np.float32)  # 将标注ann中的bbox从左上点的坐标+bbox长宽 -> 中心点的坐标+bbox的长宽
        bbox_x1y1x2y2 = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                                 np.float32)  # 将标注ann中的bbox从左上点的坐标+bbox长宽 -> 左上点的坐标+右下点的坐标

        original_image_w, original_image_h = image.size
        cx, cy, tw, th = bbox_xywh
        p = round((tw + th) / 2, 2)
        template_square_size = int(np.sqrt((tw + p) * (th + p)))  # a
        detection_square_size = int(template_square_size * 2)  # A = 2a

        # pad
        detection_lt_x, detection_lt_y = cx - detection_square_size // 2, cy - detection_square_size // 2
        detection_rb_x, detection_rb_y = cx + detection_square_size // 2, cy + detection_square_size // 2
        left = -detection_lt_x if detection_lt_x < 0 else 0
        top = -detection_lt_y if detection_lt_y < 0 else 0
        right = detection_rb_x - original_image_w if detection_rb_x > original_image_w else 0
        bottom = detection_rb_y - original_image_h if detection_rb_y > original_image_h else 0
        padding = tuple(map(int, [left, top, right, bottom]))
        padding_image_w, padding_image_h = left + right + original_image_w, top + bottom + original_image_h

        template_img_padding = ImageOps.expand(image, border=padding, fill=mean_template_and_detection)
        detection_img_padding = ImageOps.expand(image, border=padding, fill=mean_template_and_detection)

        # crop
        tl = cx + left - template_square_size // 2
        tt = cy + top - template_square_size // 2
        tr = padding_image_w - tl - template_square_size
        tb = padding_image_h - tt - template_square_size
        template_img_crop = ImageOps.crop(template_img_padding.copy(), (tl, tt, tr, tb))

        dl = np.clip(cx + left - detection_square_size // 2, 0, padding_image_w - detection_square_size)
        dt = np.clip(cy + top - detection_square_size // 2, 0, padding_image_h - detection_square_size)
        dr = np.clip(padding_image_w - dl - detection_square_size, 0, padding_image_w - detection_square_size)
        db = np.clip(padding_image_h - dt - detection_square_size, 0, padding_image_h - detection_square_size)
        detection_img_crop = ImageOps.crop(detection_img_padding.copy(), (dl, dt, dr, db))

        detection_tlcords_of_original_image = (cx - detection_square_size // 2, cy - detection_square_size // 2)
        detection_rbcords_of_original_image = (cx + detection_square_size // 2, cy + detection_square_size // 2)

        detection_tlcords_of_padding_image = (
        cx - detection_square_size // 2 + left, cy - detection_square_size // 2 + top)
        detection_rbcords_of_padding_image = (
        cx + detection_square_size // 2 + left, cy + detection_square_size // 2 + top)

        # resize
        template_img_resized = template_img_crop.copy().resize((127, 127))
        detection_img_resized = detection_img_crop.copy().resize((256, 256))

        template_resized_ratio = round(127 / template_square_size, 2)
        detection_resized_ratio = round(256 / detection_square_size, 2)

        segmentation_in_padding_img = segmentation_in_original_image
        for i in range(len(segmentation_in_original_image)):
            for j in range(len(segmentation_in_original_image[i]) // 2):
                segmentation_in_padding_img[i][2 * j] = segmentation_in_original_image[i][2 * j] + left
                segmentation_in_padding_img[i][2 * j + 1] = segmentation_in_original_image[i][2 * j + 1] + top

        x11, y11 = detection_tlcords_of_padding_image
        x12, y12 = detection_rbcords_of_padding_image

        segmentation_in_cropped_img = segmentation_in_padding_img
        for i in range(len(segmentation_in_padding_img)):
            for j in range(len(segmentation_in_padding_img[i]) // 2):
                segmentation_in_cropped_img[i][2 * j] = segmentation_in_padding_img[i][2 * j] - x11
                segmentation_in_cropped_img[i][2 * j + 1] = segmentation_in_padding_img[i][2 * j + 1] - y11
                segmentation_in_cropped_img[i][2 * j] = np.clip(segmentation_in_cropped_img[i][2 * j], 0,
                                                                x12 - x11).astype(np.float32)
                segmentation_in_cropped_img[i][2 * j + 1] = np.clip(segmentation_in_cropped_img[i][2 * j + 1], 0,
                                                                    y12 - y11).astype(np.float32)

        segmentation_in_detection = segmentation_in_cropped_img
        for i in range(len(segmentation_in_cropped_img)):
            for j in range(len(segmentation_in_cropped_img[i])):
                segmentation_in_detection[i][j] = segmentation_in_cropped_img[i][j] * detection_resized_ratio

        blcords_of_bbox_in_padding_image, btcords_of_bbox_in_padding_image, brcords_of_bbox_in_padding_image, bbcords_of_bbox_in_padding_image = \
        bbox_x1y1x2y2[0] + left, bbox_x1y1x2y2[1] + top, bbox_x1y1x2y2[2] + left, bbox_x1y1x2y2[3] + top
        blcords_of_bbox_in_detection, btcords_of_bbox_in_detection, brcords_of_bbox_in_detection, bbcords_of_bbox_in_detection = blcords_of_bbox_in_padding_image - x11, btcords_of_bbox_in_padding_image - y11, brcords_of_bbox_in_padding_image - x11, bbcords_of_bbox_in_padding_image - y11
        x1 = np.clip(blcords_of_bbox_in_detection, 0, x12 - x11).astype(np.float32)
        y1 = np.clip(btcords_of_bbox_in_detection, 0, y12 - y11).astype(np.float32)
        x2 = np.clip(brcords_of_bbox_in_detection, 0, x12 - x11).astype(np.float32)
        y2 = np.clip(bbcords_of_bbox_in_detection, 0, y12 - y11).astype(np.float32)
        cords_of_bbox_in_cropped_detection = np.array([x1, y1, x2, y2], dtype=np.float32)
        cords_of_bbox_in_resized_detection = cords_of_bbox_in_cropped_detection * detection_resized_ratio

        return (template_img_resized, detection_img_resized, cords_of_bbox_in_resized_detection, segmentation_in_detection)

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
        tbbox, _ = self._get_bbox_center_from_mask(template_mask)
        sbbox, sc = self._get_bbox_center_from_mask(detection_mask)

        # get segmentation and transformed segmentation
        contours_template, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_detect, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours_template = np.concatenate(contours_template).reshape(-1, 2)
        contours_detect = np.concatenate(contours_detect).reshape(-1, 2)

        template_target, _, _, _ = self.transform_cords(template[0], tbbox, [contours_template[::20].reshape(-1)])
        _, detection_target, bbox_of_detection, detection_seg = self.transform_cords(detection[0], sbbox, [contours_detect[::20].reshape(-1)])

        template_target = np.array(template_target)
        detection_target = np.array(detection_target)
        if len(template_target.shape) == 2:
            template_target = np.expand_dims(template_target, axis=2)
            template_target = np.concatenate((template_target, template_target, template_target), axis=-1)
            detection_target = np.expand_dims(detection_target, axis=2)
            detection_target = np.concatenate((detection_target, detection_target, detection_target), axis=-1)

        ################################################################
        # TODO 
        # 需要将template, detection, tbbox, sbbox, sc转换到各自的坐标系下，
        # 并且需要将detection_mask变换到255坐标系下
        # template和detection 是一个list, 里面的元素分别为: 原始PIL图片， PIL格式的mask, 图片路径
        ################################################################
        detection_mask_trans = np.zeros((detection_target.shape[0], detection_target.shape[1]), dtype=np.uint8)
        cv2.drawContours(detection_mask_trans, [np.array(detection_seg).reshape(-1, 2)], -1, 255, cv2.FILLED)
        kernel = np.ones((3,3), dtype=np.uint8) * 255
        detection_mask_trans = cv2.morphologyEx(detection_mask_trans, cv2.MORPH_CLOSE, kernel)

        # sc_x = np.mean(detection_mask_trans.nonzero()[1])
        # sc_y = np.mean(detection_mask_trans.nonzero()[0])
        # sc = np.array([sc_x, sc_y])

        sc = self.transform_one_point_cords(detection[0], sbbox, sc)
        sc = np.array(sc)

        # find center in feature map
        f_cx = round(max(0, (sc[0] - 31) / 8))
        f_cy = round(max(0, (sc[1] - 31) / 8))

        # operation on feature map coordinate
        valid_centers = self.get_valid_center_from_fm(25, f_cx, f_cy)
        gt_class = self.gen_gt_class(25, valid_centers)

        valid_centers_in_ori = self.coord_transform(valid_centers, 'f2o')

        contours, _ = cv2.findContours(detection_mask_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
            template_target = self.transforms(template_target)
            detection_target = self.transforms(detection_target)

        meta = {}
        meta['imgId'] = {
            'template': template[-1],
            'detection': detection[-1],
        }
        meta['template'] = template_target  # 模板 3 * 127 * 127
        meta['detection'] = detection_target  # 检测 3 * 256 * 256
        meta['bbox_of_detection'] = bbox_of_detection  # 检测帧坐标系下的bbox,
        meta['center'] = sc  # 检测帧坐标系下中心坐标
        meta['distance'] = distance  # 检测帧坐标系下距离
        meta['coords'] = new_coord  # 字典， 键为角度，键值为距离
        meta['targets'] = {
            'distances': distances,  # 25 * 25 * 36
            'gt_class': gt_class,  # 25 * 25
        }
        # meta['valid_centers'] = valid_centers
        meta['mask'] = detection_mask_trans
        # meta['ori_mask'] = detection_mask
        return meta




if __name__ == "__main__":
    # root_dir = './DAVIS/Annotations/480p/'
    # for cat in CATEGORY:

    #     fileDir = os.path.join(root_dir, cat)
    #     fileNames = os.listdir(fileDir)
    #     fileNames.sort()

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

    #         gray_mask = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
    #         contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         contours_ = np.concatenate(contours)  # size: n * 1 * 2
    #         contours_ = contours_.reshape(-1)  # 此处为opencv的x, y坐标，后处理需要交换
    #         print(contours)
    #         print(contours_)

    #         cv2.drawContours(gray_mask, contours, -1, 50, cv2.FILLED)
    #         cv2.imshow("mask", gray_mask)

    #         # contours2 = cv2.findContours(cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         # print(contours2)
    #         cv2.waitKey(0)



#########################################################################
# dataloader visualize
########################################################################
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
        print((detection.shape))
        center = temp['center'][0].numpy()
        distance = temp['distance'][0].numpy()
        coords = temp['coords']
        # print(coords)
        # GT_reg = temp['targets']['distances'][0].reshape(-1, 36)
        # GT_cls = temp['targets']['gt_class'][0]
        bbox = temp['bbox_of_detection'][0]
        GT_mask = temp['mask'][0].numpy()
        new_coords = get_contours_from_polar(center[0], center[1], coords)

        mask = cv2.UMat(np.array([GT_mask, GT_mask, GT_mask], dtype=np.uint8).transpose(1,2,0))
        print(type(mask) == type(detection))
        for i in range(len(new_coords)):
            cv2.line(detection, (int(center[0]), int(center[1])), (int(new_coords[i][0]), int(new_coords[i][1])), (0,0,255), 1)

        cv2.rectangle(detection, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1)
        cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1)

        template_show = np.zeros_like(detection)
        template_show[64:64+127, 64:64+127, :] = template

        imgs = np.hstack([template_show, detection, mask.get()])
        cv2.imshow("imgs", imgs)
        cv2.waitKey(0)

#############################################
#
#######################################