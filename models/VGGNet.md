#VGGNet的理解

##背景
VGGnet出自论文[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
在我看来VGGNet是对AlexNet的加深，论文Diss了AlexNet中提出的局部响应归一化（LRN）对性能提升并没有什么帮助


##网络配置
1. 3x3的卷积核。
整个网络均使用了3x3，stride=1的卷积核，为保证空间分辨率，each side填充一个元素，padding=1
为什么使用3x3的卷积核？
第一个好处是加深了深度，因为两个3x3的卷积核可表示一个5x5的卷积核的有效感受野，三个3x3的卷积核可表示一个7x7的卷积核的有效感受野，
第二个好处是减少了参数量，拿三层3x3的卷积核和一层7x7的卷积核为例，设输入输出的通道数均为C，3x3的参数量=3*(3*3*C*C)=27C^2，
而对于7x7的参数量=7*7*C*C=49C^2
2. 5个2x2，stride=2的MaxPooling
3. 三个全连接层（在Imgnet上，4096-4096-1000），前两个使用Dropout，最后一个不使用。在网上看到,后来发现这些全连接层即使被去除，对于性能也没有什么影响，这样就显著降低了参数数量.
4. 所有的隐藏层都带有ReLU
5. 论文中参数的设置，BatchSize=256,momentum=0.9,dropout=0.5,lr初始设定为10−2，然后当验证集准确率停止改善时，减少10倍,学习率总共降低3次
###VGGNet网络结构
![image](https://github.com/DengChan/MLStudy/raw/master/images/VGG.jpg)
###VGG19
![image](https://github.com/DengChan/MLStudy/raw/master/images/VGG19.png)


##More Details
###参数初始化
因为由于深度网络中梯度的不稳定，不好的初始化可能会阻碍学习,为了避免这个问题，作者先用了A网络训练得到的参数作为其他深层网络的初始值,
对于A，从均值为0和方差为10−2的正态分布中采样权重，偏置初始化为零。
###训练和测试的图像尺寸设置。
1. 训练
引入了多尺度训练的概念，裁剪尺度固定为224x224，但训练尺度S（指图像最小边缩放至S）可大于等于224,S的取法有两种(1)单尺度，S=256或S=384 （2）多尺度训练。S从区间[Smin,Smax]中随机取样
2. 测试
(1)图像缩放至尺寸Q，但不一定等于S
(2)全连接层转换为卷积层，FC1 -> 7x7 conv，FC2 -> 1x1 conv ， FC3 -> 1x1 conv
(3)水平翻转图像增强测试集

###对于S和Q的取法是这样理解的
有两个概念多尺度训练和多尺度评估，分别对应S,Q。
  |S单一尺度,S=S0|S多尺度(S∈[Smin,Smax])
  单尺度评估|Q=S|,Q=1/2(Smin+Smax)
  多尺度评估|Q=S-32,Q=S+32|Q = {Smin,Smax,1/2(Smin+Smax)}
  
