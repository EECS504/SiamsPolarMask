import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import cv2


class COCODataset(data.Dataset):
    def __init__(self, annFilePath, imgDir, catId=1, transform=None):
        super(COCODataset, self).__init__()
        self.annFilePath = annFilePath
        self.imgDir = imgDir
        self.catId = catId
        self.transform = transform
        # annFile = '../../504Proj/annotations/instances_val2017.json'
        self.coco = COCO(annFilePath)
        self.imgIds = self.coco.getImgIds(catIds=self.catId)

    def __getitem__(self, index):
        meta = {}

        img_info = self.coco.loadImgs(self.imgIds[index])[0]
        img = cv2.imread(self.imgDir + img_info['file_name'])

        annIds = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.catId, iscrowd=False)
        anns = self.coco.loadAnns(annIds)
        max_area_ann = max(anns, key=lambda x: x['area'])    # ann ID

        bbox = max_area_ann['bbox']
        mask = self.coco.annToMask(max_area_ann)

        c_x = np.mean(mask.nonzero()[0])            # numpy 下的x, y坐标 与opencv相反
        c_y = np.mean(mask.nonzero()[1])
        center = torch.Tensor([c_x, c_y])
        ##################################################
        # TODO
        # 添加中心点周围8个点坐标及其对应边界点
        # 考虑中心点在边界的情况
        # 输入与输出的坐标变换公式为 p_in = 8p_out + 31
        # 具体分为两个阶段 第一阶段 p_in = 8p_backbone + 7 第二阶段 p_backbone = p_out + 3
        ##################################################

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours)          # size: n * 1 * 2
        contours = contours.reshape(-1, 2)           # 此处为opencv的x, y坐标，后处理需要交换

        distance, new_coord = self.get_36_coordinates(c_x, c_y, contours)  # distance is a list, new_coord is a dictionary which keys are angles(0 ~ 360, 10)

        ####################################################
        # TODO
        # Transform image

        # meta['image'] = self.transform(img)
        ####################################################

        meta['id'] = max_area_ann['id']
        meta['imgId'] = max_area_ann['image_id']
        meta['bbox'] = bbox
        meta['center'] = center
        meta['distance'] = distance
        meta['coords'] = new_coord
        # meta['mask'] = torch.Tensor(mask).resize((255, 255))

        return meta


    def __len__(self):
        return len(self.imgIds)

    def get_36_coordinates(self, c_x, c_y, pos_mask_contour):
        # 输入为opencv坐标系下的contuor， 在计算时转化为Numpy下的坐标
        ct = pos_mask_contour
        x = torch.Tensor(ct[:, 1] - c_x)      # opencv x, y交换
        y = torch.Tensor(ct[:, 0] - c_y)

        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x ** 2 + y ** 2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]
        ct2 = ct[idx]

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
        # for idx in range(36):
        #     dist = new_coordinate[idx * 10]
        #     distances[idx] = dist

        return distances, new_coordinate

annFile = './Data/instances_val2017.json'
imgDir = './Data/val2017/'
train_data = COCODataset(annFilePath=annFile, imgDir=imgDir)
train_loader = data.DataLoader(dataset=train_data, batch_size=5, shuffle=False)

for i, Data in enumerate(train_loader):
    if i > 0:
        break
    print(Data['id'])

