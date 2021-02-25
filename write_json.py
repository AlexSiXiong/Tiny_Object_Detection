# dummy label(json) data
import json

l1 = {"images": [], "annotations": []}

img1 = {"license": 1, 'file_name': '123.jpg'}
img2 = {"license": 2, 'file_name': '456.jpg'}
l1["images"].append(img1)
l1["images"].append(img2)

anno1 = {"image_id": 289343, "bbox": [473.07, 395.93, 38.65, 28.67]}
anno2 = {"image_id": 123, "bbox": [55, 53, 55, 57]}
l1["annotations"].append(anno1)
l1["annotations"].append(anno2)

j = './t.json'
with open(j, 'w') as f:
    json.dump(l1, f, indent=4)
