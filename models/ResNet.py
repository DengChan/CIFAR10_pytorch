# -*- coding: utf-8 -*-
"""
ResNet For CIFAR10 
ResNet20/32/44/56/110
n = {3,5,7,9,18} 
n is how many times the block is repeated in a layer 
For CIFAR10,all convolutions' kernel size is 3x3
"""

import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class ResBlock(BasicModule):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv_a = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(out_channels)
        
        self.conv_b = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.downsampling = None

        if stride != 1 or in_channels != out_channels:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self,x):
        fx = self.conv_a(x)
        fx = self.bn_a(fx)
        fx = F.relu(fx,inplace = True)
        
        fx = self.conv_b(fx)
        fx = self.bn_b(fx)
        '''如果输入尺寸与输出尺寸不相同，则需要执行stride=2和1x1的卷积，统一尺寸后才能相加'''
        if self.downsampling is not None:
            x = self.downsampling(x)
        fx = nn.functional.relu(fx+x,inplace = True)
        return fx
    


class cifarResNet(BasicModule):
    def __init__(self,depth,num_class):
        super(cifarResNet,self).__init__()
        self.model_name = 'ResNet'+str(depth)
        self.depth = depth
        #深度若不符合条件则报错
        assert (self.depth-2)%6==0,'depth should be one of 20, 32, 44, 56, 110'
        #block_nums表示每一层的Block个数，即论文中的n
        self.block_nums = (self.depth-2)//6
        
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        #self.in_channels表示当前的输入维度
        self.in_channels = 16
        self.bn = nn.BatchNorm2d(self.in_channels)
        '''conv2: 输入: size 32x32x16 
                  filter: 3x3 ,16个 ,首个stride = 1 
                  输出:size 32x32x16
        '''
        self.conv2 = self._make_layer(out_channels = 16,stride = 1)
        
        '''conv3: 输入: size 32x32x16 
                  filter: 3x3 ,32个 ,首个stride = 2
                  输出:size 16x16x32
        '''
        self.conv3 = self._make_layer(32,2)
        
        '''conv4: 输入: size 16x16x32 
                  filter: 3x3 ,64个 ,首个stride = 2
                  输出:size 8x8x64
        '''
        self.conv4 = self._make_layer(64,2)
        
        '''
        以averagepool-fc-softmax结束
        avgpool kernel_size = 8 得到1x1*64的特征图像
        '''
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64,num_class)

    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.functional.relu(x,True)        
        x = self.conv2(x)        
        x = self.conv3(x)        
        x = self.conv4(x)        
        x = self.pool(x)        
        x = x.view(x.size(0),-1)        
        x = self.classifier(x)
        return x

    def _make_layer(self,out_channels,stride=1):
        downsampling = None
        if self.in_channels !=out_channels or stride != 1:
            downsampling = nn.Sequential(nn.Conv2d(self.in_channels,out_channels,kernel_size=1,stride=stride),
                                         nn.BatchNorm2d(out_channels))
        layers = []
        # 第一个Block
        layers.append(ResBlock(self.in_channels,out_channels,stride))
        # 更新in_channels
        self.in_channels = out_channels
        # 逐个加入后面的Block
        for i in range(1,self.block_nums):
            layers.append(ResBlock(out_channels,out_channels))
        
        return nn.Sequential(*layers)


cfg = {18: [2, 2, 2, 2],
       34: [3, 4, 6, 3],
       50: [3, 4, 6, 3],
       101: [3, 4, 23, 3]}


class BottleNeck(BasicModule):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsampling = None
        if stride != 1 or in_channels != out_channels:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = F.relu(fx)
        fx = self.conv2(fx)
        fx = self.bn2(fx)
        fx = F.relu(fx)
        fx = self.conv3(fx)
        fx = self.bn3(fx)
        if self.downsampling != None:
            x = self.downsampling(x)
        fx = F.relu(fx+x)
        return fx




class ResNet(BasicModule):
    def __init__(self, depth):
        super(ResNet, self).__init__()
        self.block = None
        self.num_blocks = cfg[depth]
        if depth == 18 or 34:
            self.block = ResBlock
        else:
            self.block = BottleNeck
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(self.block,self.num_blocks[0],64,1)
        self.layer2 = self.make_layer(self.block,self.num_blocks[1],128,2)
        self.layer3 = self.make_layer(self.block, self.num_blocks[2], 256, 2)
        self.layer4 = self.make_layer(self.block, self.num_blocks[3], 512, 2)
        self.linear = nn.Linear(512*self.block.expansion, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def make_layer(self, block, num_block, out_channels, stride):
        strides = [stride]+[1]*(num_block-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels,out_channels,s))
            self.in_channels = block.expansion*out_channels
        return nn.Sequential(*layers)


def resnet18():
    return ResNet(18)


def resnet34():
    return ResNet(34)


def resnet50():
    return ResNet(50)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    


        


