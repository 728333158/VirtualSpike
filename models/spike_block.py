from models.spike_layer import SpikeConv, tdBatchNorm2d, SpikePool, SpikeModule, myBatchNorm3d, LIFNode
import torch.nn as nn
import math
from models.resnet import BasicBlock


class SpikeBasicBlock(SpikeModule):
    """
    Implementation of Spike BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, step=2):
        super().__init__()
        self.step = step
        self.conv1 = SpikePool(basic_block.conv1, step=step)
        self.bn1 = myBatchNorm3d(basic_block.bn1, step=step)
        self.relu1 = LIFNode(step)

        self.conv2 = SpikePool(basic_block.conv2, step=step)
        self.bn2 = myBatchNorm3d(basic_block.bn2,step=step)
        if basic_block.downsample is None:
            self.downsample = None
        else:
            if len(basic_block.downsample) == 3:
                self.downsample = nn.Sequential(
                    SpikePool(basic_block.downsample[0], step=step),
                    SpikePool(basic_block.downsample[1], step=step),
                    myBatchNorm3d(basic_block.downsample[2], step=step)
                )
            else:
                self.downsample = nn.Sequential(
                    SpikePool(basic_block.downsample[0], step=step),
                    myBatchNorm3d(basic_block.downsample[1], step=step)
                )
        self.output_act = LIFNode(step)
        self.stride = basic_block.stride

    def forward(self, s):
        temp, x = s
        x = super().forward(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out1 = self.output_act(out)
        return out, out1


def is_normal_blk(module):
    return isinstance(module, BasicBlock)


def is_spike_blk(module):
    return isinstance(module, SpikeBasicBlock)


specials = {BasicBlock: SpikeBasicBlock}
