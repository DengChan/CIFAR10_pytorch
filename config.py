# -*- coding: utf-8 -*-
"""
Default Config
"""

class DefaultConfig(object):
    model_names = ['vgg16','vgg16_bn','vgg19_bn','vgg19','resnet20','resnet32'
                   ,'resnet44','resnet56','resnet110',]
    use_gpu = True
    model = 'resnet20'
    load_model_path = ''
    epoch = 350
    lr = 0.01
    lr_decay = 0.1
    weight_decay = 5e-4
    batch_size = 128
    momentum = 0.9
    num_workers = 4
    print_freq = 20
    
    def parse(self,kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        
        #print config information
        for k,v in self.__class__.__dict__.items():
            if not k[0:2]=='__':
                print(k,getattr(self,k))

cfg = DefaultConfig() 
