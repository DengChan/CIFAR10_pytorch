# -*- coding: utf-8 -*-
"""
ResNet For CIFAR10 
ResNet20/32/44/56/110
n = {3,5,7,9,18} 
n is how many times the block is repeated in a layer 
For CIFAR10,all convolutions' kernel size is 3x3
"""

import torch.nn as nn
from .BasicModule import BasicModule


class ResBlock(BasicModule):
    def __init__(self,in_channels,out_channels,stride=1,downsampling=None):
        super(ResBlock,self).__init__()
        self.conv_a = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn_a = nn.BatchNorm2d(out_channels)
        
        self.conv_b = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn_b = nn.BatchNorm2d(out_channels)
        
        self.downsampling = downsampling
    
    def forward(self,x):
        fx = self.conv_a(x)
        fx = self.bn_a(fx)
        fx = nn.functional.relu(fx,inplace = True)
        
        fx = self.conv_b(fx)
        fx = self.bn_b(fx)
        '''如果输入尺寸与输出尺寸不相同，则需要执行stride=2和1x1的卷积，统一尺寸后才能相加'''
        if self.downsampling is not None:
            x = self.downsampling(x)
        fx = nn.functional.relu(fx+x,inplace = True)
        return fx
    


class ResNet(BasicModule):
    def __init__(self,depth,num_class):
        super(ResNet,self).__init__()
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
        if self.in_channels!=out_channels or stride != 1:
            downsampling = nn.Sequential(nn.Conv2d(self.in_channels,out_channels,kernel_size=1,stride=stride),
                                         nn.BatchNorm2d(out_channels))
        layers = []
        #第一个Block
        layers.append(ResBlock(self.in_channels,out_channels,stride,downsampling))
        #更新in_channels
        self.in_channels = out_channels
        #逐个加入后面的Block
        for i in range(1,self.block_nums):
            layers.append(ResBlock(out_channels,out_channels))
        
        return nn.Sequential(*layers)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    


        


