import numpy as np
from rbbox_overlaps import rbbx_overlaps
from iou_cpu import get_iou_matrix


boxes1 = np.array([[50, -50, 100, 900, 0],
                  [60, 60, 100, 200, 0]], np.float32)

boxes2 = np.array([[50, -50, 100, 900, -8.],
                  [200, 200, 100, 200, 0.]], np.float32)


riou_gpu = rbbx_overlaps(boxes1, boxes2, 0)
print(riou_gpu)

riou_cpu = rbbx_overlaps(boxes1, boxes2)
print(riou_cpu)



