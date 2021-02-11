import torch
import json
import numpy as np
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO

json_file = 'val2017.json'
img_root = './val2017'
coco = COCO(json_file)  # 读取json信息
images = coco.getImgIds()    # 获取所有图片id
counter = 0
for img_id in images:
    img_info = coco.loadImgs(ids=[img_id])[0]
    print(img_info)
    file_name = coco.loadImgs(ids=[img_id])[0]['file_name']  #  获取一张图片文件名
    img_path = os.path.join(img_root, file_name)                  #  图片绝对路径
    # ann_ids =coco.getAnnIds(imgIds=[img_id])                    #  获取这张图片下所有box的id
    ann_ids = coco.getAnnIds(imgIds=img_info["id"], iscrowd=False)
    anns = coco.loadAnns(ids=ann_ids)


    print(file_name)
    print(img_path)
    print(ann_ids)
    for ann in anns:
        print(ann['bbox'])
        print(ann["category_id"])



    print('-------------------------')
    counter += 1
    if counter == 1:
        break

category_ids = coco.getCatIds()
print(category_ids)

# 根据类别id获取所有类别的名称
categories = [c['name'] for c in coco.loadCats(category_ids)]
# 类别名称与id的相互映射
category_to_id_map = dict(zip(categories, category_ids))
id_to_category_map = dict(zip(category_ids, categories))
classes = ['__background__'] + categories
num_classes = len(classes)
print(num_classes)
print(category_ids)
print(categories)
print(category_to_id_map)

json_category_id_to_contiguous_id = {
    v: i + 1
    for i, v in enumerate(coco.getCatIds())
}
# 类别索引与其id的映射
contiguous_category_id_to_json_id = {
    v: k
    for k, v in json_category_id_to_contiguous_id.items()
}
print(json_category_id_to_contiguous_id)
print(contiguous_category_id_to_json_id )
