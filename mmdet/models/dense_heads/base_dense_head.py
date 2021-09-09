from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.

        Args:
            x:                        (list[Tensor])  经过 FPN 后的 features
            img_metas:                  (list[dict])  一个 batch 的 image 的信息的 list, 如: 大小, 缩放等.
            gt_bboxes:                (list[Tensor])  一个 batch 的 Ground truth bboxes 的 list,
                                                      每个图片 gt bboxes 的形状为 (num_gts, 4).
            gt_labels:         (list[Tensor] | None)  一个 batch 的 Ground truth labels 的 list,
                                                      每个图片 gt labels 的形状为 (num_gts,).
            gt_bboxes_ignore:  (list[Tensor] | None)  一个 batch 忽略的 ground truth bboxes,
                                                      每个图片的 gt_bboxes_ignore 的形状为 (num_ignored_gts, 4).
            proposal_cfg:              (mmcv.Config)  测试 / 后处理 的配置, 如果为 None, 就会使用 test_cfg

        Returns:
            tuple:
                losses:          (dict[str, Tensor])  一个 loss 字典
                proposal_list:        (list[Tensor])  一个批次的图片的 proposal 列表,
                                                      每个 proposal 的形状为 (1000, 5)
        """
        # 前向计算, 拿到 confidence 和 bbox 坐标偏移的结果.
        # 结果为 [cls_list, reg_list]
        # cls_list 是每个尺度的分类预测结果, 形状为 [torch.Size([batch, anchor, H, W]), ...]
        # reg_list 是每个尺度的回归预测结果, 形状为 [torch.Size([batch, anchor × 4, H, W ]), ...]
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # 计算损失
        # 结果为 dict('loss_rpn_cls', 'loss_rpn_bbox')
        # loss_rpn_cls  是每个尺度的分类损失, 是一个 tensor 数值.
        # loss_rpn_bbox 是每个尺度的回归损失, 是一个 tensor 数值.
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # proposal_cfg 为 None, 只返回损失不提供 proposal, 例如: 单独训练 RPN 就不需要提供 proposal
        if proposal_cfg is None:
            return losses
        # proposal_cfg 不是 None, 返回损失且提供 proposal
        else:
            # 一个批次的图片的 proposal 列表, 每个 proposal 的形状为 (1000, 5)
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
