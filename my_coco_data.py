import json
import pandas as pd
import numpy as np
import os
import cv2

# file_name, class, xmin, ymin, xmax, ymax

input = pd.read_csv('./try/test.csv')

for row in input.iterrows():
    row = list(row)[1]

    file_name = row[0]
    object_class = row[1]
    x_min = row[2]
    y_min = row[3]
    w = row[4] - row[2]
    h = row[5] - row[3]

