# YOLOv3 Network Slimming

[Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)

[Rethinking The Smaller-Norm-Lessinformative](https://arxiv.org/abs/1802.00124?context=cs)

## Require
pytorch 0.41
window 10

## Fix cfg
* Change `num=9` to `num=3` for each **yolo layer** .
* `mask` must order by ascending.
* `anchors` must include integer only.
* set `random` as 1.

## How to use
1.Train with channel sparsity regularization.
```bash
python sparsity_train.py -sr --s 0.0001 --image_folder coco.data --cfg yolov3.cfg --weights yolov3.weights
```
2.Prune channels with small scaling factors.
```bash
python new_prune.py --cfg yolov3.cfg --weights checkpoints/yolov3_sparsity_100.weights --percent 0.3
```
3.Fine-tune the pruned network.
```bash
python sparsity_train.py --image_folder coco.data  --cfg prune_yolov3.cfg --weights prune_yolov3.weights 
```

