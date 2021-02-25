import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
# from pprint import pprint as print

def draw_box(fig_, bbox, c):
    return fig_.axes.add_patch(plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3],
        fill=False, edgecolor=c, linewidth=2))


def draw_text(fig_, bbox, text, c):
    return fig_.axes.text(bbox[0] + 6, bbox[1] - 7,
                          text,
                          horizontalalignment='left',
                          verticalalignment='bottom',
                          fontsize='x-small',
                          backgroundcolor=c)


def drawing(fig_, bbox, annotation=''):
    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
    draw_box(fig_, bbox, c)
    if len(annotation) != 0:
        draw_text(fig_, bbox, annotation, c)


def serialize_category_id_dic(original_ids):
    # json_category_id_to_contiguous_id
    json_category_id_to_contiguous_id = {
        v: i
        for i, v in enumerate(original_ids)
    }
    return json_category_id_to_contiguous_id



json_file = 'val2017.json'
img_root = './val2017'
coco = COCO(json_file)  # 读取json信息
images = coco.getImgIds()  # 获取所有图片id
counter = 0
category_ids = coco.getCatIds()


# 根据类别id获取所有类别的名称
categories = [c['name'] for c in coco.loadCats(category_ids)]
print(categories)
contiguous_id_dic = serialize_category_id_dic(coco.getCatIds())
print("xxxxxxxxxxxxxxxxxxxxxxx")

for img_id in images:
    img_info = coco.loadImgs(ids=[img_id])[0]
    print(img_info)
    file_name = coco.loadImgs(ids=[img_id])[0]['file_name']  # 获取一张图片文件名
    img_path = os.path.join(img_root, file_name)  # 图片绝对路径
    # ann_ids =coco.getAnnIds(imgIds=[img_id])  #  获取这张图片下所有box的id
    ann_ids = coco.getAnnIds(imgIds=img_info["id"], iscrowd=None)
    anns = coco.loadAnns(ids=ann_ids)

    print(file_name)
    print(img_path)
    print(ann_ids)
    img = Image.open(img_path)
    fig = plt.imshow(img)

    for ann in anns:
        bbox = ann['bbox']
        categories_id = ann["category_id"]
        obj_name = categories[contiguous_id_dic[categories_id]]
        drawing(fig, bbox, obj_name)

    plt.show()
    print('-------------------------')
    counter += 1
    if counter == 1:
        break

# 类别名称与id的相互映射
category_to_id_map = dict(zip(categories, category_ids))
id_to_category_map = dict(zip(category_ids, categories))
print(id_to_category_map)
print(category_to_id_map)
classes = ['__background__'] + categories
num_classes = len(classes)
print(num_classes)
print(category_ids)

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
