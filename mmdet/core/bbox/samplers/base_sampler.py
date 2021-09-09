from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers.
    Args:
        num:                  (int)   采样后的总数
        pos_fraction        (float)   正样本的比例
        neg_pos_ub            (int)   负样本与正样本数量比值的上界
        add_gt_as_proposals  (bool)   是否添加 ground truth 作为 proposal.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        """正样本采样, 此方法需要子类实现."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        """负样本采样, 此方法需要子类实现."""
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.
        采样正负样本

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Args:
            assign_result:  (:obj:`AssignResult`):  Bbox 分配后的结果.
            bboxes:         (Tensor):               需要采样的 bbox, 形状为 (N, 4)
            gt_bboxes:      (Tensor):               Ground truth bboxes, 形状为 (n_gts, 4)
            gt_labels:      (Tensor, optional):     ground truth bboxes 的类别标签.

        Returns:
            :obj:`SamplingResult`:  采样的结果.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        # 是否将 gt 添加到正样本.
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        # 预期的正样本数量（eg, rpn 128）
        num_expected_pos = int(self.num * self.pos_fraction)
        # 采样正样本.
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        # 正样本不够的用负样本填充
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
