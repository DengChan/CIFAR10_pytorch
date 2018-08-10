1.  cfg.parse()时 for k,v in kwargs.item() :  忘记加item()
2. for i,data in enumerate(train_dataloader)  忘记加enumerate
3. softmax 位指明维度
4: BasicModule 初始化参数时 对m.bias ，要先判断是否为None


ResNet:
1. 当x与fx维度不同时，对x进行处理时，用1x1，s=2的卷积核