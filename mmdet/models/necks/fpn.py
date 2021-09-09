import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels:             (List[int])     每个尺度的输入通道数, 也是 backbone 的输出通道数.
        out_channels:                  (int)     fpn 的输出通道数, 所有尺度的输出通道数相同, 都是一个值.
        num_outs:                      (int)     输出 stage 的个数.(可以附加额外的层, num_outs 不一定等于 in_channels)
        start_level:                   (int)     使用 backbone 的起始 stage 索引, 默认为 0.
        end_level:                     (int)     使用 backbone 的终止 stage 索引。
                                                 默认为 -1, 代表到最后一层(包括)全使用.
        add_extra_convs:        (bool | str)     可以是 bool 或 str:
                                                (bool)  bool 代表是否添加额外的层.(默认值: False)
                                                        True:   在最顶层 feature map 上添加额外的卷积层,
                                                                具体的模式需要 extra_convs_on_inputs 指定.
                                                        False:  不添加额外的卷积层
                                                (str)   str  需要指定 extra convs 的输入的 feature map 的来源
                                                        'on_input':     最高层的 feature map 作为 extra 的输入
                                                        'on_lateral':   最高层的 lateral 结果 作为 extra 的输入
                                                        'on_output':    最高层的经过 conv 的 lateral 结果作为 extra 的输入
        extra_convs_on_inputs:  (bool, deprecated)  True  等同于 `add_extra_convs='on_input'
                                                    False 等同于 `add_extra_convs='on_output'
                                                    默认值为 True
        relu_before_extra_convs:      (bool)     是否在 extra conv 前使用 relu. (默认值: False)
        no_norm_on_lateral:           (bool)     是否对 lateral 使用 bn. (默认值: False)
        conv_cfg:                     (dict)     构建 conv 层的 config 字典. (默认值: None)
        norm_cfg:                     (dict)     构建  bn  层的 config 字典. (默认值: None)
        act_cfg:                      (dict)     构建 activation  层的 config 字典. (默认值: None)
        upsample_cfg:                 (dict)     构建 interpolate 层的 config 字典. (默认值: `dict(mode='nearest')`)
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels          # [256, 512, 1024, 2048]
        self.out_channels = out_channels        # 256
        self.num_ins = len(in_channels)         # 4
        self.num_outs = num_outs                # 5
        self.relu_before_extra_convs = relu_before_extra_convs  # False
        self.no_norm_on_lateral = no_norm_on_lateral            # False
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # end_level 是对 backbone 输出的尺度中使用的最后一个尺度的索引
        # 如果是 -1 表示使用 backbone 最后一个 feature map, 作为最终的索引.
        if end_level == -1:
            self.backbone_end_level = self.num_ins      # 4
            # 因为还有 extra conv 所以存在 num_outs > num_ins - start_level 的情况
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            # 如果 end_level < inputs, 说明不使用 backbone 全部的尺度, 并且不会提供额外的层.
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level                      # 0
        self.end_level = end_level                          # -1
        self.add_extra_convs = add_extra_convs              # False
        assert isinstance(add_extra_convs, (str, bool))
        # add_extra_convs 可以是 bool 或 str
        # 1. add_extra_convs 是 str
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            # 确保 add_extra_convs 是 'on_input', 'on_lateral' 或 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        # 2. add_extra_convs 是 bool, 需要看 extra_convs_on_inputs
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 构建 lateral conv 和 fpn conv
        for i in range(self.start_level, self.backbone_end_level):
            # 水平卷积: 1×1, C=256,
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            # fpn 输出卷积: 3×3, C=256, P=1
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        # 只有 add_extra_convs 为 True 或 str 时才添加 extra_convs
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                # extra conv 是 3x3 步长为 2, padding 为 1 的卷积
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        # ====================== 进行水平计算(1x1卷积) ====================
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # ========================== 计算 top-down =============================
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # 因为range函数不包括右边的端点, 所以可以使用 i - 1
                laterals[i - 1] += F.interpolate(laterals[i],
                                              **self.upsample_cfg)
            # 没有 scale 的情况, 需要计算下层的 feature map 大小.
            else:
                # 计算下层 feature map 大小
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        # ========================== 计算输出的结果 =============================
        # part 1: 计算所有 lateral 的输出的结果
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            # 使用 max pool 获得更高层的输出信息, 如: Faster R-CNN, Mask R-CNN (4 lateral + 1 max pool)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            # 添加额外的卷积层获得高层输出信息, 如: RetinaNet (3 lateral + 2 conv3x3 stride2)
            else:
                # 'on_input':   最高层的 feature map 作为 extra 的输入
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                # 'on_lateral': 最高层的 lateral 结果 作为 extra 的输入
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                # 'on_output':  最高层的经过 conv 的 lateral 结果作为 extra 的输入
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                # 计算 input extra
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
