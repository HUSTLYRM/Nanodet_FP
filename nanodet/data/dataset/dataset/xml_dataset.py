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

import logging
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

from pycocotools.coco import COCO

from .coco import CocoDataset


def get_file_list(path, type=".xml"):
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext == type:
                file_names.append(filename)
    return file_names

# CocoXML类，xml格式
class CocoXML(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, "annotation file format {} not supported".format(
            type(dataset)
        )
        self.dataset = dataset
        self.createIndex()


# 数据处理相关内容，XMLDataset，继承了CocoDataset
class XMLDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(XMLDataset, self).__init__(**kwargs)

    # 将xml格式转换成coco格式
    def xml_to_coco(self, ann_path):
        """
        convert xml annotations to coco_api
        :param ann_path:
        :return:
        """
        logging.info("loading annotations into memory...")
        tic = time.time()
        ann_file_names = get_file_list(ann_path, type=".xml")
        logging.info("Found {} annotation files.".format(len(ann_file_names)))
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
        ann_id = 1
        for idx, xml_name in enumerate(ann_file_names):     # 将所有数据转换
            tree = ET.parse(os.path.join(ann_path, xml_name))
            root = tree.getroot()
            file_name = root.find("filename").text
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            info = {                                        # 组装图片的info
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": idx + 1,
            }
            image_info.append(info)
            for _object in root.findall("object"):          # 每张图片的所有object加入到数据
                category = _object.find("name").text
                if category not in self.class_names:
                    logging.warning(
                        "WARNING! {} is not in class_names! "
                        "Pass this box annotation.".format(category)
                    )
                    continue
                for cat in categories:
                    if category == cat["name"]:
                        cat_id = cat["id"]
                xmin = int(_object.find("bndbox").find("xmin").text)        # voc 格式bbox两个点，左上点和右下点
                ymin = int(_object.find("bndbox").find("ymin").text)
                xmax = int(_object.find("bndbox").find("xmax").text)
                ymax = int(_object.find("bndbox").find("ymax").text)
                w = xmax - xmin                                             # coco 格式需要的是x, y, w, h, 在这里进行处理
                h = ymax - ymin

                # TODO 新增points部分[x1, y1, x2, y2, x3, y3, x4, y4]   
                x1 = int(_object.find("points").find("x1").text)
                y1 = int(_object.find("points").find("y1").text)
                x2 = int(_object.find("points").find("x2").text)
                y2 = int(_object.find("points").find("y2").text)
                x3 = int(_object.find("points").find("x3").text)
                y3 = int(_object.find("points").find("y3").text)
                x4 = int(_object.find("points").find("x4").text)
                y4 = int(_object.find("points").find("y4").text)

                if w < 0 or h < 0:                                          # 不合适的数据
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(xml_name)
                    )
                    continue

                coco_box = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]  # 组装成coco数据bbox
                
                points = [x1, y1, x2, y2, x3, y3, x4, y4]       # TODO 新增points部分数据
        
                ann = {
                    "image_id": idx + 1,    
                    "bbox": coco_box,       # 重点关注  coco_box
                    "points": points,       # 重点关注  TODO 新增points部分
                    "category_id": cat_id,  # 重点关注  类别
                    "iscrowd": 0,
                    "id": ann_id,
                    "area": coco_box[2] * coco_box[3],      # w*h   面积
                }

                annotations.append(ann)
                ann_id += 1

        # 组装成一个数据集的coco_dict, 图片info的列表, 类别列表, annotations列表
        coco_dict = {
            "images": image_info,
            "categories": categories,   
            "annotations": annotations,
        }
        logging.info(
            "Load {} xml files and {} boxes".format(len(image_info), len(annotations))
        )
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        
        # 将组装好的coco_dict数据, 返回
        return coco_dict

    # 获取数据info
    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.xml_to_coco(ann_path)
        self.coco_api = CocoXML(coco_dict)              # 更改
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
