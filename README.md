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
Boxes array 1:
[[ 50. -50. 100. 900.   0.]
 [ 60.  60. 100. 200.   0.]]
Boxes array 2:
[[ 50. -50. 100. 900.  -8.]
 [200. 200. 100. 200.   0.]]
Computing rotated iou with gpu...
[[0.5245546  0.        ]
 [0.20191085 0.        ]]
Computing rotated iou with cpu...
[[0.5245546  0.        ]
 [0.20191085 0.        ]]
```

