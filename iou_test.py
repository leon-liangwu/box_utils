import numpy as np
from rbbox_overlaps import rbbx_overlaps
from iou_cpu import get_iou_matrix
import time


boxes1 = np.array([[50, -50, 100, 900, 0],
                  [60, 60, 100, 200, 0]], np.float32)

boxes2 = np.array([[50, -50, 100, 900, -8.],
                  [200, 200, 100, 200, 0.]], np.float32)


boxes1 = np.tile(boxes1, (30000, 1))

print('Boxes array 1 shape :')
print(boxes1.shape)

print('Boxes array 2 shape :')
print(boxes2.shape)

print('Computing rotated iou with gpu 1st time...')
t_start = time.time()
riou_gpu = rbbx_overlaps(boxes1, boxes2, 0)
t_end = time.time()
print('gpu time: ', t_end - t_start)

print('Computing rotated iou with gpu 2nd time...')
t_start = time.time()
riou_cpu = rbbx_overlaps(boxes1, boxes2)
t_end = time.time()
print('gpu time: ', t_end - t_start)

print('Computing rotated iou with gpu 3rd time...')
t_start = time.time()
riou_cpu = rbbx_overlaps(boxes1, boxes2)
t_end = time.time()
print('gpu time: ', t_end - t_start)

print('Computing rotated iou with cpu 1st time...')
t_start = time.time()
riou_cpu = get_iou_matrix(boxes1, boxes2)
t_end = time.time()
print('cpu time: ', t_end - t_start)

print('Computing rotated iou with cpu 2nd time...')
t_start = time.time()
riou_cpu = get_iou_matrix(boxes1, boxes2)
t_end = time.time()
print('cpu time: ', t_end - t_start)
