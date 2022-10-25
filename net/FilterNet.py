import torch
import numpy as np
import math
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
def dataSet(txtPath):
    with open(txtPath,mode="r") as f:
        lines=f.readlines()
        datas=[]
        labels=[]
        for line in lines:
            line=line.strip().split(" ")
            line=list(map(float,line))
            datas.append(line[:3])
            labels.append([line[3]])
    return np.array(datas),np.array(labels)
# ----------------------------------------------------------------------------------
# 自注意力机制
# ----------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        x=torch.unsqueeze(x,dim=2)
        # print("x shape",x.shape)
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # get attention scorce
        attention_scores = torch.matmul(q, k.permute(0,2,1))*self._norm_fact
        # 计算soft-max
        attention_scores_softmax = F.softmax(attention_scores, dim=-1)
        outputs = torch.matmul(attention_scores_softmax, v)
        outputs=torch.squeeze(outputs,dim=2)
        return outputs
# ----------------------------------------------------------------------------------
# mlp注意力机制
# ----------------------------------------------------------------------------------
class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=1,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2,n_hidden3,n_output):
        super(Net, self).__init__()
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        self.att = SelfAttention(1, 1, 1)
        # 三层感知机
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)

    def forward(self, x):
        x=self.att(x)
        h1=self.hidden1(x)
        h2=self.hidden2(h1)
        h3=self.hidden3(h2)
        h3 = F.relu(h3)
        out = self.predict(h3)
        return out
class mlpNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2,n_hidden3,n_output):
        super(mlpNet, self).__init__()
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        # 三层感知机
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)

        self.predict = torch.nn.Linear(n_hidden3, n_output)

        self.ea = ExternalAttention(d_model=3, S=64)
    def forward(self, x):
        x1=self.ea(x)
        x=x1+x
        h1=self.hidden1(x)
        h2=self.hidden2(h1)
        h3=self.hidden3(h2)
        h3 = F.relu(h3)
        out = self.predict(h3)
        return out
# ----------------------------------------------------------------------------------
# Focal Loss
# ----------------------------------------------------------------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self,alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, yPred, yTrue):
        eps = 1e-7
        ce = torch.mul(-1 * torch.log(yPred + eps),yTrue)
        focalLoss = torch.pow((1 - yPred), self.gamma) * ce
        focalLoss=torch.mul(focalLoss,self.alpha)
        focalLoss = torch.sum(focalLoss, dim=1)
        focalLoss=torch.mean(focalLoss)
        return focalLoss
class  StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self,X):
        '''根据训练数据集X获得数据的均值和方差'''
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
        return self
    def transform(self,X):
        '''将X根据Standardcaler进行均值方差归一化处理'''
        resX = np.empty(shape=X.shape,dtype=float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col]-self.mean_[col]) / (self.scale_[col])
        return resX

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # 训练
    # ----------------------------------------------------------------------------------
    StandardScaler=StandardScaler()
    txt="trainData.txt"
    features,labels=dataSet(txt)
    features[:,:2]=features[:,:2]*100
    print("1",sum(labels==1))
    print("0", sum(labels == 0))
    # StandardScaler.fit(features)
    # features=StandardScaler.transform(features)
    index0 = (labels == [0])
    index0 = np.squeeze(index0, axis=1)
    labels0 = np.array(labels[index0])
    # 因为标签为0的数量远远大于标签为1的数据，所以随机抽取m个标签为0的数据作为训练
    labels0 = labels0[:61510]
    index1 = (labels == [1])
    index1 = np.squeeze(index1, axis=1)
    labels1 = labels[index1]
    features0 = np.array(features[index0])
    features0 = features0[:61510]
    features1 = features[index1]
    features = np.concatenate([features0, features1], axis=0)
    labels = np.concatenate([labels0, labels1], axis=0)
    np.random.seed(300)
    np.random.shuffle(features)
    np.random.seed(300)
    np.random.shuffle(labels)
    features=torch.tensor(features,dtype=torch.float32)
    labels=torch.tensor(labels,dtype=torch.long)
    trainfeatures=features[:-1000]
    trainlabels=labels[:-1000]
    trainlabels = torch.squeeze(trainlabels, dim=1)
    # 将标签转换为独热码
    # trainonehotlabels = torch.zeros(trainlabels.shape[0], 2).long()
    # trainonehotlabels.scatter_(dim=1, index=trainlabels.long(), src=torch.ones(trainlabels.shape[0], 2).long())
    # trainonehotlabels=trainonehotlabels.float()
    # # 将张量转换为可训练
    # trainonehotlabels=Variable(trainonehotlabels)
    testfeatures=features[-1000:]
    testlabels=labels[-1000:]
    net = Net(3, 512, 256, 128, 2)  # 输入节点6个，输出节点2个
    # 采用SGD优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 采用交叉熵损失函数
    lossFunc = torch.nn.CrossEntropyLoss()
    maxaccuracy = 0
    print(trainlabels.shape)
    for t in range(20000):
        optimizer.zero_grad()
        prediction = net(trainfeatures)
        # softmax_1 = nn.Softmax(dim=1)
        # trainpre=softmax_1(prediction)
        loss = lossFunc(prediction,trainlabels)
        print("loss:", loss)
        loss.backward()
        optimizer.step()
        n = testfeatures
        pre = net(n)
        softmax_2 = nn.Softmax(dim=1)
        pre = softmax_2(pre)
        prelabels = pre.argmax(dim=1)
        print("预测的结果:", prelabels)
        print("真实的结果:", testlabels.reshape(testlabels.shape[0]))
        accuracy = float(sum(testlabels.reshape(testlabels.shape[0]) == prelabels)) / float(prelabels.shape[0])
        print("第%d次迭代的准确率:%f" % (t, accuracy))
        # 保存权重参数
        if accuracy >= maxaccuracy:
            maxaccuracy = accuracy
            print("save model")
            # 保存模型语句
            torch.save(net.state_dict(), "./weights/2A20000.pth")
    # ----------------------------------------------------------------------------------
    # 测试
    # ----------------------------------------------------------------------------------
    # txt = ".\\filterTest.txt"
    # features, testlabels = dataSet(txt)
    # features=torch.tensor(features,dtype=torch.float32)
    # print(features)
    # net = Net(3, 512, 256, 128, 2)  # 输入节点6个，输出节点2个
    # net.load_state_dict(torch.load('./filter.pth'))
    # pre = net(features)
    # softmax_2 = nn.Softmax(dim=1)
    # pre = softmax_2(pre)
    # prelabels = pre.argmax(dim=1)
    # print("预测的结果:", prelabels)
    # print("真实的结果:", testlabels.reshape(testlabels.shape[0]))