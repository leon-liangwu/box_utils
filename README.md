# box_utils
box utils, including iou and nms of bbox and rbbox

a pure version from https://github.com/yangJirui/RRPN_FPN_Tensorflow/tree/master/libs/box_utils

#### build
```
python setup.py build_ext --inplace
```

#### usage
```
python iou_test.py
```
#### output
```
Boxes array 1 shape :
(60000, 5)
Boxes array 2 shape :
(2, 5)
Computing rotated iou with gpu 1st time...
('gpu time: ', 0.22462105751037598)
Computing rotated iou with gpu 2nd time...
('gpu time: ', 0.0014929771423339844)
Computing rotated iou with gpu 3rd time...
('gpu time: ', 0.0014219284057617188)
Computing rotated iou with cpu 1st time...
('cpu time: ', 0.44385290145874023)
Computing rotated iou with cpu 2nd time...
('cpu time: ', 0.4446260929107666)
```

