import argparse

import mmcv
import torch

from utils import reformat_data
from mmdet.core import bbox2result
from img_show import show_result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMSeg to TorchScript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--img-path', help='path of test img', default=None)
    parser.add_argument('--show', action='store_true', help='show TorchScript graph')
    parser.add_argument('--verify', action='store_true', help='verify the TorchScript model')
    parser.add_argument('--model-file', type=str, default='tmp.tr')
    parser.add_argument('--shape', type=int, nargs='+', default=[512, 512],
                        help='input image size (height, width)')
    args = parser.parse_args()
    return args


def postprocess(outputs, num_classes):
    num_imgs = len(outputs['bboxes'])
    bbox_results = [bbox2result(outputs['bboxes'][i],
                                outputs['labels'][i],
                                num_classes)
                    for i in range(num_imgs)]
    print("type:",type(outputs))
    if 'segm' in outputs:  # 分割
        im_mask = outputs['segm'][0]
        labels = outputs['labels'][0]
        cls_segms = [[] for _ in range(num_classes)]  # BG is not included in num_classes
        for i in range(len(labels)):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy() > 0)

        results = list(zip(bbox_results, [cls_segms]))
    else:
        results = list(zip(bbox_results))
    return results


def main(args, cfg):
    img = mmcv.imread(args.img_path)
    mm_inputs = reformat_data(img, cfg)
    model = torch.jit.load(args.model_file)

    outputs = model(mm_inputs['img'], mm_inputs['img_metas'])
    # results = postprocess(outputs, cfg.model.roi_head.mask_head.num_classes)
    results = postprocess(outputs, 80)
    show_result(img, results,
                score_thr=0.3,
                bbox_color='green',
                text_color='green',
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                wait_time=0,
                out_file="mask.jpg")


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    main(args, cfg)
