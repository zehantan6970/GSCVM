import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
def dataSet(txtPath):
    with open(txtPath,mode="r") as f:
        lines=f.readlines()
        datas=[]
        labels=[]
        for line in lines:
            line=line.strip().split(" ")
            line=list(map(float,line))
            datas.append(line[:3])
            labels.append(line[3])
    return np.array(datas),np.array(labels)
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2,n_hidden3,n_output):
        super(Net, self).__init__()
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        # 三层感知机
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)

    def forward(self, x):
        h1=self.hidden1(x)
        h2=self.hidden2(h1)
        h3=self.hidden3(h2)
        h3 = F.relu(h3)
        out = self.predict(h3)
        return out
if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # 训练
    # ----------------------------------------------------------------------------------
    txt=".\\matchedFilterData.txt"
    features,labels=dataSet(txt)
    np.random.seed(30)
    np.random.shuffle(features)
    np.random.seed(30)
    np.random.shuffle(labels)
    features=torch.tensor(features,dtype=torch.float32)
    labels=torch.tensor(labels,dtype=torch.long)
    trainfeatures=features[:-30]
    trainlabels=labels[:-30]
    testfeatures=features[-30:]
    testlabels=labels[-30:]
    net = Net(3, 512, 256, 128, 2)  # 输入节点6个，输出节点2个
    # 采用SGD优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 采用交叉熵损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    maxaccuracy = 0
    for t in range(20000):
        optimizer.zero_grad()
        prediction = net(trainfeatures)
        loss = loss_func(prediction, trainlabels)
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
        accuracy = (sum(testlabels.reshape(testlabels.shape[0]) == prelabels)) / prelabels.shape[0]
        print("第%d次迭代的准确率:%f" % (t, accuracy))
        # 保存权重参数
        if accuracy >= maxaccuracy:
            maxaccuracy = accuracy
            print("save model")
            # 保存模型语句
            torch.save(net.state_dict(), "./pth/filter20000.pth")
    # ----------------------------------------------------------------------------------
    # 测试
    # ----------------------------------------------------------------------------------
    # txt = "C:\\Users\\asus\\Desktop\\mydata\\filterTest.txt"
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