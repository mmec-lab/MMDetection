import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class AnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Args:
        num_classes:              (int)   类别的个数, 不包括背景类.
        in_channels:              (int)   输入的 feature map 的通道数.
        feat_channels:            (int)   中间提取特征使用的通道数.
        anchor_generator:        (dict)   anchor generator 的配置文件字典.
        bbox_coder:              (dict)   box coder 的配置文件字典.
        reg_decoded_bbox:        (bool)   如果为 True, 将会对解码的 bbox 回归损失. (默认值: False).
        background_label:  (int | None)   背景标签的 id. 在 RPN 中为 0, 其他的 head 为 num_classes.
                                          如果为 None 会自动设置为 num_classes.
        loss_cls:                (dict)   分类 loss 的配置文件字典.
        loss_bbox:               (dict)   回归 loss 的配置文件字典.
        train_cfg:               (dict)   anchor head 的训练配置.
        test_cfg:                (dict)   anchor head 的测试配置.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layers='Conv2d', std=0.01)):
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels        # 256, FPN 每个尺度的输出通道数
        self.num_classes = num_classes        # 类别个数, 不包括背景类.
        self.feat_channels = feat_channels    # 256
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        # 使用 FocalLoss 不需要对 proposal 进行采样, 所以 sampling = False
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # 只有训练时才会分配正负样本 (assigner) 和平衡正负样本的数量 (sampler)
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        # anchor 的生成, 无论是训练或测试都需要.
        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        # num_anchors 每个尺度 base_anchor 的数量.
        # 通常每个尺度 base_anchor 的数量相同, 如 RPN(3, 3, 3, 3, 3), 除了 ssd.
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        """初始化 Head 的 layer"""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        对单尺度的 feature map 前向传播.
        Args:
            x (Tensor):    单尺度的 feature map

        Returns:
            tuple:
                cls_score (Tensor): 单尺度的置信度 (通道数为: anchors 的数量 * num_classes)
                bbox_pred (Tensor): 单尺度预测的偏移量 (通道数为: anchors 的数量 * 4)
        """
        # 所有尺度使用相同的 conv_cls 和 conv_reg 进行预测.
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.
        前向传播, 获得网络预测的分类和回归的结果.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.

        Args:
            feats:  (tuple[Tensor]): 经过 backbone 和 neck 后的 features 的元祖, 每个元素是一个尺度的 feature.

        Returns:
            tuple: 一个元祖, 包括分类分数和 bbox 回归预测的结果
                cls_scores (list[Tensor]):  每个尺度的分类分数的 list, 每个元素代表一个尺度, 数据类型为 tensor.
                                            每个元素的形状为 [batch, anchor 数量 × 类别个数, H, W]
                bbox_preds (list[Tensor]):  每个尺度的 bbox 回归的 list, 每个元素代表一个尺度, 类型为 tensor.
                                            每个元素的形状为 [batch, anchor 数量 × 4, H, W]
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.
        根据特征图的大小获取 anchor.


        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.

        Args:
            featmap_sizes: (list[tuple]):           各个尺度的 feature map 的大小.
            img_metas:     (list[dict]):            一个批次的图像属性信息
            device:        (torch.device | str):    返回的 tensor 的设备

        Returns:
            tuple:
                anchor_list:        (list[list[Tensor]]): 一个批次图片的 anchor, 每个元素是一张图片所有尺度的 anchor
                valid_flag_list:    (list[list[Tensor]]): 一个批次图片有效的 anchor, 每个元素是一张图片所有尺度的 anchor
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        # 由于一个批次所有图像的特征图大小相同, 所以只用生成 anchor 一次, 就可以得到整个批次的 anchor.
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        # 因为同一个 batch 的图片大小相同所以这里直接循环 batch 次数次 anchor 就行.
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        # 对于每个图像, 计算多尺度 anchor 的有效标志.
        # 形状为 list(list(Tensor)), 其中 Tensor 代表一张图片一个尺度的 Anchor.
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.
        计算一张图片 anchor 的回归和分类的目标

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images

        Args:
            flat_anchors:       (Tensor):   合并后的多尺度的 anchor. 形状为: (num_anchors ,4).
            valid_flags:        (Tensor):   合并后的多尺度的 anchor 的 flag, 形状为 (num_anchors,).
            gt_bboxes:          (Tensor):   图像的 ground truth bbox, 形状为 (num_gts, 4).
            gt_bboxes_ignore:   (Tensor):   需要忽略的 Ground truth bboxes 形状为: (num_ignored_gts, 4).
            img_meta:           (dict):     此图像的属性信息
            gt_labels:          (Tensor):   每个 box 的 Ground truth labels, 形状为 (num_gts,).
            label_channels:     (int):      label 所在的通道.
            unmap_outputs:      (bool):     是否将输出映射回原始 anchor 配置.

        Returns:
            tuple:
                labels:          (Tensor)     训练的标签, 形状为 (anchor 总数,)
                label_weights:   (Tensor)     训练标签的权重, 形状为 (anchor 总数,)
                bbox_targets:    (Tensor)     bbox 训练的目标值, 形状为 (anchor 总数, 4)
                bbox_weights:    (Tensor)     bbox 训练目标值的权重, 形状为 (anchor 总数, 4)
                pos_inds:        (Tensor)     正样本的索引, 形状为 (正样本总数,)
                neg_inds:        (Tensor)     负样本的索引, 形状为 (负样本总数,)
        """
        # ===================== 1. 筛选出有效的 anchor ===========================
        # 获得有效的 flag, 这里的 inside_flags 就等于 valid_flags, 形状为 (num_anchors,)
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        # 如果 anchor 没有一个有效, 直接返回
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        # 筛选有效的 anchor, 此时 Anchor 数量会减少为有效的 anchor 数量.
        anchors = flat_anchors[inside_flags, :]

        # ========================== 2. anchor 分配正负样本 ==============================
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)

        # ========================== 3. anchor 正负样本采样 ==============================
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        # ======================== 4. 构建 label 和 bbox 的目标和权重 =====================
        # 有效的 anchor 数量
        num_valid_anchors = anchors.shape[0]
        # bbox 目标, 初始化将目标设置为 0
        bbox_targets = torch.zeros_like(anchors)
        # bbox 权重, 即是否需要算入损失, 是否需要网络学习. 初始化将权重设置为 0
        bbox_weights = torch.zeros_like(anchors)
        # label 的目标, 初始化先将所有有效的 anchor 的标签标记为背景 (0)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        # label 的权重, 初始化将将权重权设置为 0
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        # 获得正负样本的索引
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # ================ （1）构建 bbox 的目标和权重 ====================
            # 获得所有正样本 box 的 anchor, 形状 [正样本数量, 4]
            if not self.reg_decoded_bbox:
                # 将 anchor 编码为中心点坐标，宽和高的偏移量
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            # 将正样本对应的 indices 设置为编码后的 anchor, 将权重设置为 1
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            # ================ （2）构建 label 的目标和权重 ===================
            if gt_labels is None:
                # 只有 rpn 的 gt_labels 才设置为 None
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                # 否则设置为对应的类别编号
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            # 将正样本的权重设置为 1
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        # ===================== 5. 填充 anchor 到没有筛选 valid flag 的长度. ==================
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            # 填充 labels
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            # 填充 label_weights
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            # 填充 bbox_targets
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            # 填充 bbox_weights
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):

        """Compute regression and classification targets for anchors in
        multiple images.
        获得一个批次的训练和回归目标.
        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end

        Args:
            anchor_list:            (list[list[Tensor]])    所有批次所有尺度的 anchor 的列表,
                                                            每个 tensor 代表一张图片的一个尺度的 anchor.
                                                            形状为 (num_anchors, 4).
            valid_flag_list:        (list[list[Tensor]]):   所有批次所有尺度 anchor 的 valid flag,
                                                            每个 tensor 代表一张图片的一个尺度的 anchor 的 valid flag.
                                                            形状为 (num_anchors,)
            gt_bboxes_list:         (list[Tensor]):         一个 batch 的 gt bbox, 每个 tensor 的形状为 (num_gts, 4)
            img_metas:              (list[dict]):           一个 batch 的图片的属性信息.
            gt_bboxes_ignore_list:  (list[Tensor]):         需要忽略的 gt bboxes
            gt_labels_list:         (list[Tensor] | None):  一个 batch 的 gt labels.
            label_channels:         (int):                  标签的通道
            unmap_outputs:          (bool):                 是否填充 anchor 到没有筛选 valid flag 的长度

        Returns:
            tuple:
                labels_list:        (list[Tensor]):     每个尺度的 label, 每个元素的形状为 (batch, n_anchors)
                label_weights_list: (list[Tensor]):     每个尺度 label 的权重, 每个元素的形状为 (batch, n_anchors)
                bbox_targets_list:  (list[Tensor]):     每个尺度的 bbox, 每个元素的形状为 (batch, n_anchors, 4)
                bbox_weights_list:  (list[Tensor]):     每个尺度 bbox 的权重, 每个元素的形状为 (batch, n_anchors, 4)
                num_total_pos:      (int):              一个批次所有图片的正样本总数
                num_total_neg:      (int):              一个批次所有图片的负样本总数
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        # 计算 batch 的数量
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        # 计算每个尺度 anchor 的数量 [187200, 46800, 11700, 2925, 780]
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        # 遍历每个图片, 合并每个图片中所有尺度的 anchor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            # 并所有尺度的 anchor合
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            # 合并所有尺度的 flag
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        # 统计所有 image 的正负样本
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.
        计算单个尺度的损失.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        Args:
            cls_score:      (Tensor): 单尺度的 box score, 形状 (batch, n_anchors * n_classes, H, W).
            bbox_pred:      (Tensor): 单尺度 bbox 的修正量, 形状 (batch, n_anchors * 4, H, W)
            anchors:        (Tensor): 单个尺度的 anchor, 形状为 (batch, n_anchors, 4).
            labels:         (Tensor): 每个 anchor 的标签, 形状为 (batch, n_anchors)
            label_weights:  (Tensor): label 的权重, 形状为 (batch, n_anchors)
            bbox_targets:   (Tensor): bbox 的修正量, 形状为 (batch, n_anchors, 4).
            bbox_weights:   (Tensor): bbox 修正量的权重, 形状为 (batch, n_anchors, 4).
            num_total_samples  (int): 如果采样, 则 num_total_samples 等于锚点总数, 否则为正样本数量。

        Returns:
            loss_cls:   (Tensor)    分类的损失值
            loss_bbox:  (Tensor)    回归的损失值
        """
        # classification loss
        # 分类损失
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # torch.Size([batch, n_anchors × 类别数, H, W]) -->  torch.Size([batch × H × W × n_anchor, 类别数])
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        # 对正负样本计算分类损失, 因为最后要相加, 所以这里 avg_factor=正负样本总数作为分子.
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        # 回归损失
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        # 只计算正样本的回归损失
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.
        计算 Head 的损失

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        Args:
            cls_scores:  (list[Tensor])  多个尺度的预测的置信度 list,
                                         其中每个尺度的 tensor 的形状为 [batch, n_anchors × 类别数, H, W]
            bbox_preds:  (list[Tensor])  多个尺度位置的修正量的 list,
                                         其中每个尺度的 tensor 的形状为 [batch, n_anchors × 4, H, W]
            gt_bboxes:   (list[Tensor])  一个 batch 每张图片的 ground truth. list 的长度为 batch 长度.
                                         每个 tensor 的形状为 (num_gts, 4) 其中维度 1 代表 [tl_x, tl_y, br_x, br_y]
            gt_labels:   (list[Tensor])  每个 gt box 的类别索引.
            img_metas:  (list[dict]):    一个批次的图像的属性信息
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: 损失的字典
        """
        # 获取各个尺度 feature map 大小
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        # 获取一个批次的 anchor 和 valid flag
        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # 得到训练的 target
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        # labels_list:  多个尺度的 list, 每个尺度的 tensor 形状 [batch, n_anchors]
        # label_weights_list: 多个尺度的 list, 每个尺度的 tensor 形状 [batch, n_anchors]
        # bbox_targets_list:  多个尺度的 list, 每个尺度的 tensor 形状 [batch, n_anchors, 4]
        # bbox_weights_list:  多个尺度的 list, 每个尺度的 tensor 形状 [batch, n_anchors, 4]
        # num_total_pos: 正样本总数
        # num_total_neg: 负样本总数
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 计算样本总数, 如果不采样, 总数为正样本个数. 否则为正负样本总个数
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        # 获得每个尺度的 anchor 数量
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        # 把 anchor 变成含多个尺度的 list
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.
        将网络的输出转化为一个批次的预测

        Args:
            cls_scores (list[Tensor]): Box scores for each level in the
                feature pyramid, has shape
                (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each
                level in the feature pyramid, has shape
                (N, num_anchors * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.

        Args:
            cls_scores:     (list[Tensor]):         每个尺度的 bbox 分数预测, 形状为 (batch, n_anchors * n_classes, H, W)
            bbox_preds:     (list[Tensor]):         每个尺度的 bbox 修正量, 形状为 (batch, n_anchors * 4, H, W)
            img_metas:      (list[dict]):           一个批次的图像属性信息
            cfg:            (mmcv.Config | None):   Test / postprocessing 配置文件, 如果为 None, 将会使用 test_cfg
            rescale:        (bool):                 如果为 True, 则返回原始图像空间中的框, 默认值：False.

        Returns:
            list(Tensor):   一个批次每个图片的 proposal 的列表, 每个 Tensor 的形状为 (nms_post, 5), 
                            其中前四列代表解码后的 bbox 坐标, 最后一列代表置信度.
        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        elif torch.jit.is_tracing():
            img_shapes = []
            for i in range(cls_scores[0].shape[0]):
                img_shapes.append(img_metas[i]['img_shape'])
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list

    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a batch item into bbox predictions.

        Args:
            mlvl_cls_scores (list[Tensor]): Each element in the list is
                the scores of bboxes of single level in the feature pyramid,
                has shape (N, num_anchors * num_classes, H, W).
            mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                bboxes predictions of single level in the feature pyramid,
                has shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of single level in feature pyramid, has shape
                (num_anchors, 4).
            img_shapes (list[tuple[int]]): Each tuple in the list represent
                the shape(height, width, 3) of single image in the batch.
            scale_factors (list[ndarray]): Scale factor of the batch
                image arange as list[(w_scale, h_scale, w_scale, h_scale)].
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1),
            device=mlvl_cls_scores[0].device,
            dtype=torch.long)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            # Always keep topk op for dynamic input in onnx
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)

                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if not self.use_sigmoid_cls:
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                  batch_mlvl_scores.shape[1],
                                                  1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
