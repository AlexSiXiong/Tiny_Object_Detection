# convert the xml coordinates to json coordinates
# so that COCO library can process them

import json
import pandas as pd
import numpy as np
import os
from random import choice
from string import digits
import json
import glob
import xml.etree.ElementTree as ET

result = {"images": [], "annotations": [], "categories": []}
id_arr = []
category_dic = {}

init_id = 1


def id_generator():
    code = list()
    for i in range(6):
        code.append(choice(digits))
    return int(''.join(code))


def unique_id_generator():
    while True:
        cur_id = id_generator()
        if cur_id not in id_arr:
            break
    id_arr.append(cur_id)
    return cur_id


def process_one_img(file_name, h, w, image_id):
    return {"file_name": file_name,
            "height": h,
            "width": w,
            "id": image_id}


def process_one_annotation(image_id, bbox, c_id, obj_id):
    return {"image_id": image_id,
            "bbox": bbox,
            "category_id": c_id,
            "id": obj_id}


def process_one_category(category_id, category_name):
    return {"id": category_id,
            "name": category_name}


def class_name_mapping_category_id(name):
    keys = category_dic.keys()
    if name not in keys:
        category_dic[name] = len(keys) + 1


def xml_reader(path):
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_id = unique_id_generator()
        object_id = unique_id_generator()
        # 1.images
        file_name = root.find('filename').text
        height = 0
        width = 0
        for i in root.findall('size'):
            height = i[1].text
            width = i[0].text
        img_info = process_one_img(file_name, height, width, image_id)

        # 2.annotation
        bbox = []
        class_name = ''
        for member in root.findall('object'):
            x = int(member[4][0].text)
            y = int(member[4][1].text)
            w = int(member[4][2].text) - x
            h = int(member[4][3].text) - y
            bbox = [x, y, w, h]

            class_name = member[0].text
            class_name_mapping_category_id(class_name)
        anno_info = process_one_annotation(image_id, bbox, category_dic[class_name], object_id)

        result["images"].append(img_info)
        result["annotations"].append(anno_info)
    # 3.category
    for k, v in category_dic.items():
        category_info = process_one_category(v, k)
        result["categories"].append(category_info)


def main():
    image_path = '/test1/'
    xml_reader(image_path)
    with open('../t.json', 'w') as f:
        json.dump(result, f, indent=4)
    print('Successfully converted xml to csv.')


main()
# {
#     "images": [],
#     "annotations": [],
#     "categories": []
# }
#
# # image
# {
#     "id": int,
#     "width": int,
#     "height": int,
#     "file_name": str,
# }
#
# # annotation
# {
#     "id": int,
#     "image_id": int,
#     "category_id": int,
#     "area": float,
#     "bbox": [x, y, width, height],
#     "iscrowd": 0 or 1,
# }

# # categories
# {
#     "id":1,
#     "name":"person",
# }
