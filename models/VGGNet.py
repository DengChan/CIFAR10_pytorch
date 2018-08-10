
"""

VGG16,VGG19


"""

import torch as t
from torch import nn
from .BasicModule import BasicModule


model_cfg={
     'vgg16':[64,64,'m',128,128,'m',256,256,256,'m',512,512,512,'m',512,512,512,'m'],
     'vgg16_bn':[64,64,'m',128,128,'m',256,256,256,'m',512,512,512,'m',512,512,512,'m'],
     'vgg19':[64,64,'m',128,128,'m',256,256,256,256,'m',512,512,512,512,'m',512,512,512,512,'m'],
     'vgg19_bn':[64,64,'m',128,128,'m',256,256,256,256,'m',512,512,512,512,'m',512,512,512,512,'m'],
     }


class VGGNet(BasicModule):
    def __init__(self,vgg_name):
        super(VGGNet,self).__init__()
        
        self.model_name = vgg_name
        
        #whether use BatchNormal or not
        isBN = False
        if 'bn' in vgg_name:
            isBN = True
        
        self.features=self._make_layers(model_cfg[vgg_name],isBN)
        
        self.classifier = nn.Sequential(
                nn.Linear(512,512),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(512,512),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(512,10),
                )
        
        
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    
        
    def _make_layers(self,model_cfg,isBN):
        layers = []
        in_channels = 3
        for x in model_cfg:
            if x == 'm':
                layers += [nn.MaxPool2d(kernel_size=2,stride = 2)]
            else:
                conv = nn.Conv2d(in_channels,x,kernel_size=3,padding=1)
                if isBN:
                    layers += [conv,nn.BatchNorm2d(x),nn.ReLU(True)]
                else:
                    layers += [conv,nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)
            
            