## Export Models for Deployment

### Features

| Model                      | ONNX | tracing | Scripting |
| -------------------------- | ---- | ------- | --------- |
| Cascade Mask RCNN with FPN | ❌    | ✅       | ❌         |
| Cascade RCNN with FPN      | ❌    | ✅       | ❌         |
| Dynamic RCNN with FPN      | ❌    | ✅       | ❌         |
| Mask RCNN with FPN         | ❌    | ✅       | ❌         |
| Faster RCNN with FPN       | ❌    | ✅       | ❌         |


### Install

```shell
cd mmdetection
pip install -v -e .
```

 ### How to Export
```
1. Two new files have been added to MMDetection to export the TORCH Script model of RCNN and validate the exported Torch Script model
2. pytorch2torchscript.py
    Args:
        description='Convert RCNN to TorchScript'
        config, help='test config file path'
        --checkpoint, help='checkpoint file', default=None
        --output-file, help= 'save torch script model file', default='tmp.tr'
        --img-path, help='path of test img', default=None
        --shape, help='input image size (height, width)'
        --verify, action='store_true', help='verify the TorchScript model'
3. demo_verify_torchscript.py
    Args:
        description='Convert RCNN to TorchScript'
        config, help='test config file path'
        --checkpoint, help='checkpoint file', default=None
        --model-file, help= 'torch script model file', default='tmp.tr'
        --img-path, help='path of test img', default=None
        --shape, help='input image size (height, width)'
```

### Export model in tracing mode

```shell
cd tools/deployment

python pytorch2torchscript.py \
/mmdetection/configs/xxx_rcnn/xxx_rcnn_r50_fpn_1x_coco.py \
--checkpoint=/mmdetection/xxx_rcnn_r50_fpn_1x_coco.pth \
--output-file=/mmdetection/xxx_rcnn_r50_fpn_1x_coco.tr \
--img-path=/mmdetection/tools/deployment/test_img.jpg \
--shape 640 480 \
--verify

### Verify

If you want to check the outputs of the converted model, *demo_verify_torchscript.py* visualizes the results and record in *torchsript_res.jpg*.
```shell
cd tools/deployment

cd mmdection/tools/deployment
python demo_verify_torchscript.py \
/mmdetection/configs/xxx_rcnn/xxx_rcnn_r50_fpn_1x_coco.py \
--model-file=/mmdetection/xxx_rcnn_r50_fpn_1x_coco.tr \
--img-path=/mmdetection/tools/deployment/test_img.jpg \
--shape 640 480
