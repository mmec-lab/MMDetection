import mmcv
import torch

from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def reformat_data(img, cfg):
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    # just get the actual data from DataContainer
    img_metas = data['img_metas'][0].data[0][0]

    _img_metas = {'ori_shape'     : torch.IntTensor(img_metas['ori_shape']),
                  'img_shape'     : torch.IntTensor(img_metas['img_shape']),
                  'pad_shape'     : torch.IntTensor(img_metas['pad_shape']),
                  'scale_factor'  : torch.Tensor(img_metas['scale_factor']),
                  'mean'          : torch.Tensor(img_metas['img_norm_cfg']['mean']),
                  'std'           : torch.Tensor(img_metas['img_norm_cfg']['std']),
                  'flip'          : torch.tensor(False),
                  'flip_direction': torch.zeros(0) if img_metas['flip_direction'] else torch.ones(
                      1),
                  }
    data['img_metas'] = _img_metas
    data['img'] = data['img'][0]
    return data
