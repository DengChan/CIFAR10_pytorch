
"""
BasicModule
"""

import torch as t
from torch import nn
import time


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) #Default model name
    
    #Initiate parameters
    def init_pm(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,0,0.0001)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            if isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
    
    #save the model as 'modelname_time.pth'
    def save(self,name=None):
        if name == None:
            prefix = 'checkpoints/'+self.model_name + '_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict,name)
        return name
        
    #load parameters from saved model
    def load(self,path):
        self.load_state_dict(t.load(path))
