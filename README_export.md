## Export Models for Deployment

### Features

| Model                      | ONNX | tracing | Scripting |
| -------------------------- | ---- | ------- | --------- |
| Cascade Mask RCNN with FPN | ❌    | ✅       | ❌         |
| Cascade RCNN with FPN      | ❌    | ✅       | ❌         |

### Install

```shell
cd mmdetection
pip install -v -e .
```

 ### How to Export

- Export model in tracing mode

```shell
cd mmdection/tools/deployment

python pytorch2torchscript.py \
/Users/wutao/Projects/CATL/work_dir/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco.py \
--checkpoint=/Users/wutao/Projects/CATL/work_dir/cascade_mask_rcnn_r50_fpn_1x_coco/latest.pth \
--output-file=/Users/wutao/Projects/CATL/work_dir/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco.tr \
--img-path=test_img.jpg \
--shape 416 608 \
--verify
```

### Verify

If you want to check the outputs of the converted model, *demo_verify_torchscript.py* visualizes the results and record in *torchsript_res.jpg*.

```shell
python demo_verify_torchscript.py \
/Users/wutao/Projects/CATL/work_dir/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco.py \
--model-file=/Users/wutao/Projects/CATL/work_dir/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco.tr \
--img-path=test_img.jpg \
--shape 416 608
```



