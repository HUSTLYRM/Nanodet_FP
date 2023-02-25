# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from .base import BaseDataset

# 定义COCO数据集类，CocoDataset继承了BaseDataset
class CocoDataset(BaseDataset):
    
    # 获取数据集的一些相关信息，这个函数只需要知道根据ann_path得到img_info就可以了（初始化调用了）
    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url':
              'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())                        
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}    
        self.cats = self.coco_api.loadCats(self.cat_ids)                         
        self.class_names = [cat["name"] for cat in self.cats]                    
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    # 获取每张图片的信息元组tuple
    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]   # data_info 就是上面那个函数的返回img_info, 在父类中调用了get_data_info方法来初始化       
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]
        if not isinstance(id, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id}
        return info

    # 获取图片的annotation，每个图片的重点标注的信息， 重点关注bbox、category_id    （points是我自己有其他需求新增的信息）
    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []          # gt就是ground truth，可以理解为就是正确的，就是标注的正样本，方便后续计算损失函数等
        gt_labels = []
        gt_points = []          # TODO 新增读取points部分，存储gt_points

        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []

        for ann in anns:            # 对于每一个ann都处理
            x1, y1, w, h = ann["bbox"]                  # 这里说明，coco格式的 ann文件中存放的是 左上角坐标以及宽高
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]             # 将bbox更新成左上角、右下角,  bbox存放的是左上点，右下点
            if ann.get("iscrowd", False) or ann.get("ignore", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)                          # 将bbox加入到gt_bboxes中
                
                x1, y1, x2, y2, x3, y3, x4, y4 = ann["points"]  # TODO 新增 points 的 部分， 在annotation中新增points
                points = [x1, y1, x2, y2, x3, y3, x4, y4]
                gt_points.append(points)                        # 将points加入到gt_points中

                gt_labels.append(self.cat2label[ann["category_id"]])    # 将类别id加入到gt_labels中
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann["keypoints"])

        if gt_bboxes:       # 列表  转换成  numpy形式的数组
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_points = np.array(gt_points, dtype=np.float32)            # TODO 新增gt_points
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_points = np.zeros((0, 8), dtype=np.float32)               # TODO 新增gt_points
            gt_labels = np.array([], dtype=np.int64)
        
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        
        # 要重要的ann组装成字典格式
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, points=gt_points    # TODO 新增gt_points
        )

        if self.use_instance_mask:
            annotation["masks"] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation["keypoints"] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation["keypoints"] = np.zeros((0, 51), dtype=np.float32)
        
        return annotation   # 将得到的重要信息返回

    # `训练时` 的关键调用，    根据idx索引来获取训练数据（将重要信息组装成了meta（dict类型））
    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.get_per_img_info(idx)               # 根据idx图片的信息
        file_name = img_info["file_name"]
        image_path = os.path.join(self.img_path, file_name) # 利用join生成图片的具体路径
        img = cv2.imread(image_path)                        # 读取图片
        if img is None:
            print("image {} read failed.".format(image_path))
            raise FileNotFoundError("Cant load image! Please check image path!")
        
        ann = self.get_img_annotation(idx)                  # 获取idx索引对应的annotaion
        
        # meta很重要, 基本就是用来计算损失函数等等的 ground truth 部分
        # TODO meta增加gt_points，组装成meta数据
        meta = dict(
            img=img,
            img_info=img_info,
            gt_bboxes=ann["bboxes"],
            gt_labels=ann["labels"],
            gt_points=ann["points"],
            gt_bboxes_ignore=ann["bboxes_ignore"],
        )

        # print(meta)

        # TODO  注意：这里处理的时候，通过查看meta的不同信息，可以看出来bbox是发生了缩放的，也要注意要对points也要进行处理，否则meta中的points数据是不合适的
        # 自此，改好了数据读取部分，预处理部分

        if self.use_instance_mask:
            meta["gt_masks"] = ann["masks"]
        if self.use_keypoint:
            meta["gt_keypoints"] = ann["keypoints"]
        input_size = self.input_size
        if self.multi_scale:
            input_size = self.get_random_size(self.multi_scale, input_size)
        meta = self.pipeline(self, meta, input_size)                # 这里对其进行了处理, 最终是调用了warp.py里面的函数，对meta中的数据进行了resize

        # print(meta)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        return meta

    # 验证时获取数据
    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)
