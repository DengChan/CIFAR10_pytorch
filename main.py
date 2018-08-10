
"""
main.py

"""

import torch as t
from torch import nn
from torch import optim 
import torchvision as tv
import torchvision.transforms as transforms

from config import cfg
import models


def main(**kwargs):
    
    '''update config'''
    cfg.parse(kwargs)
    
    '''step 1. define net'''
    assert cfg.model in cfg.model_names,'No such a model!Check your inputs.'
    net = None
    if cfg.model[0:3] == 'vgg':
        net = models.VGGNet(cfg.model)
    elif cfg.model[0:3] == 'res':
        net = models.ResNet(int(cfg.model[6:]),10)
    else:
        pass
        
    #初始化权重
    net.init_pm()
    #net = VGGNet('vgg19')
    if cfg.use_gpu:
        net = net.cuda()
    
    '''step 2. load data'''
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    
    
    #训练集
    train_tf=transforms.Compose([
            transforms.RandomCrop(34,4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    
    train_dataset = tv.datasets.CIFAR10(root='data/',train=True,download=True,
                                        transform = train_tf)
    train_dataloader = t.utils.data.DataLoader(train_dataset,batch_size=cfg.batch_size,
                                               shuffle = True,num_workers = cfg.num_workers)
    
    #测试集
    test_tf=transforms.Compose([
            transforms.ToTensor(),
            ])
    test_dataset = tv.datasets.CIFAR10(root='data/',train=False,download=True,transform=test_tf)
    test_dataloader = t.utils.data.DataLoader(test_dataset,batch_size = cfg.batch_size,
                                              shuffle = False,num_workers=cfg.num_workers)
    
    #定义误差函数和优化函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=cfg.lr,momentum=0.9,weight_decay =1e-4)  
    lr = cfg.lr
    
    #记录验证集的最高精度
    best_accuracy = 0.0
    for epoch in range(0,cfg.epoch):
        total=0
        correct=0
        total_loss = 0
        
        for i,data in enumerate(train_dataloader):
            inputs,labels = data
            if cfg.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            if cfg.use_gpu :
                outputs = outputs.cuda()
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _,predicted = t.max(outputs.data,1)
            #统计信息
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            if i % (cfg.print_freq)==cfg.print_freq-1 :
                print('epoch %d,batch %d:  Loss:%0.3f ' % (epoch+1,i+1,total_loss/(i+1)))
        
        print('epoch %d, accuracy:%0.3f%%' % (epoch+1,100.*correct/total))
        
        #在测试集上验证精度
        valid_accuracy = valid(net,test_dataloader)
        if valid_accuracy>best_accuracy:
            best_accuracy = valid_accuracy
            net.save()
        
        #更新lr
        if epoch == 100 or epoch == 150:
            lr = lr * cfg.lr_decay
            for pmg in optimizer.param_groups:
                pmg['lr'] = lr
            
        
            
        
def valid(net,dataloader):
    total = 0
    correct = 0
    #转换为验证模式
    net.eval()
    for i,data in enumerate(dataloader):
        inputs,labels = data
        if cfg.use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        
        _,predicted = t.max(outputs.data,1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    net.train()
    print('valid_accuracy : %.3f%%' % (100.*correct/total))
    #print('correct:%d | total %d' % (correct,total))
    return 1.0*correct/total
    
    
    
    
if __name__ == '__main__':
    import fire
    fire.Fire()
    
    
    
    
    
    
    
    
    
    