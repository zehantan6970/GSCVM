import numpy as np
import math
from net.FilterNet import Net,mlpNet
import torch
from collections import Counter
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from itertools import combinations,combinations_with_replacement
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def singletxt2arr(path):
    """
    path:存储点云n个三维坐标点的txt
    output: 转换为数组存储，output=[[x,y,z],...]
    """
    with open(path,mode="r") as f:
        lines= f.readlines()
        output=[]
        for i, line in enumerate(lines):
            line = line.strip().split(",")
            lst=np.array(list(map(float,line)))
            output.append(lst)
    # output:dtype=list
    return output
def superGluetxt2arr(path):
    """
    读取superGlue生成的txt文件,txt文件内的数据shape=(n,6),6是(x1,y1,z1,x2,y2,z2),分别为源点云与目标点云中对齐的点的xyz
    ------------------------------------------------------------------------------------
    param:
        path:存储点云n个三维坐标点的txt
    return:
        lpsn: 转换为数组存储，lpsn=[[x,y,z],...]
        lptn: 转换为数组存储，lptn=[[x,y,z],...]
        del_id: 多个点对到一个点与不存在深度信息的索引
    """
    with open(path,mode="r") as f:
        lines= f.readlines()
        lpsn=[]
        lptn=[]
        del_id=[]
        lpsnArr=[]
        lptnArr=[]
        lpsnRepeat=[]
        lptnRepeat=[]
        for i, line in enumerate(lines):
            line = line.split(" ")
            lst = list(map(float, line))
            if lst[:3] in lpsnArr and lst[:3] not in lpsnRepeat:
                lpsnRepeat.append(lst[:3])
            elif lst[3:6] in lptnArr and lst[3:6] not in lptnRepeat:
                lptnRepeat.append(lst[3:6])
            else:
                lpsnArr.append(lst[:3])
                lptnArr.append(lst[3:6])
        for i, line in enumerate(lines):
            line = line.split(" ")
            lst=list(map(float,line))
            if 0 in lst[:6] or lst[3:6] in lptnRepeat or lst[:3] in lpsnRepeat:
                del_id.append(i)
            else:
                lpsn.append(lst[:3])
                lptn.append(lst[3:6])
    # output:dtype=list
    return np.array(lpsn),np.array(lptn),np.array(del_id)
def get_FDH(scope, dist): #scope统计范围，dist距离矩阵

    FDH_mat = np.zeros((dist.shape[0], int(scope/.2)+1))
    for i in range(dist.shape[0]): #遍历每个点
        for k in range(dist.shape[1]):
            judge = int(dist[i][k]/.2)
            if dist[i][k] >= scope:
                FDH_mat[i][int(scope/.2)] += 1
            else:
                FDH_mat[i][judge] += 1
    for mat in FDH_mat:
        mat /= np.sum(mat)
    return FDH_mat

#计算巴氏距离
def get_Bhadist(hist_s, hist_t):#hits--source点CDF
    Bhadist = 0
    for i in range(hist_s.shape[0]):
            Bhadist += np.sqrt(hist_s[i] * hist_t[i])
    #print(Bhadist)
    return Bhadist

#根据CDF进行匹配
def matchBYbhd(fdhs, fdht):
    match_matrix = []
    for i in range(len(fdhs)):
        count_mat = []
        for j in range(len(fdht)):
            count_mat.append(get_Bhadist(fdhs[i], fdht[j]))
        #print(count_mat)
        if max(count_mat)<0.3:
            index = math.inf
        else:
            index = count_mat.index(max(count_mat))
        match_matrix.append([i, index])
    return match_matrix
def get_distance(points):#points--n*3的矩阵
    dist = np.zeros((points.shape[0],points.shape[0]))
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            if i == j:
                continue
            else:
                dist[i][j] = np.sqrt(pow(points[i][0]-points[j][0], 2)+pow(points[i][1]-points[j][1], 2)
                                     +pow(points[i][2]-points[j][2], 2))
    return dist
def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)
def predict(features):
    """
     使用神经网络判断对齐的点是否符合阈值要求
     ------------------------------------------------------------------------------------
     features: (distDiv,distDiv,angleDiv)
     return: 返回预判断为对齐的点是否符合阈值要求
     """
    features = torch.tensor(features, dtype=torch.float32)
    net = Net(3, 512, 256, 128, 2) # 输入节点6个，输出节点2个
    net.load_state_dict(torch.load('/home/light/gree/align/Hypothesis/net/weights/2A20000.pth'))
    pre = net(features)
    softmax_2 = nn.Softmax(dim=1)
    pre = softmax_2(pre)
    prelabels = pre.argmax(dim=1)
    return prelabels
def isMatchA(index,engineeringlpsn,engineeringlptn,*args):
    """
    纯计算，仅采用距离角度的差值
    ------------------------------------------------------------------------------------
    index: 预判断是否对齐点的索引
    engineeringlpsn: 源点云的n对点
    engineeringlptn: 目标点云的n对点
    rgs: 默认为对齐的点的索引
    return: 返回是否符合阈值要求
    """
    matchedA=args[0][0]
    matchedB=args[0][1]
    matchedsA=np.array(engineeringlpsn[matchedA])
    matchedtA=np.array(engineeringlptn[matchedA])
    matchedsB=np.array(engineeringlpsn[matchedB])
    matchedtB=np.array(engineeringlptn[matchedB])
    s=np.array(engineeringlpsn[index])
    t=np.array(engineeringlptn[index])
    vectorsA=np.array(s-matchedsA).reshape([1,3])
    vectorsB = np.array(s - matchedsB).reshape([3, 1])
    vectortA=np.array(t-matchedtA).reshape([1,3])
    vectortB = np.array(t - matchedtB).reshape([3, 1])
    subA=abs(abs(np.sqrt(np.sum(vectorsA*vectorsA)))-abs(np.sqrt(np.sum(vectortA*vectortA))))
    subB = abs(abs(np.sqrt(np.sum((vectorsB* vectorsB)))) - abs(np.sqrt(np.sum(vectortB * vectortB))))
    coss=vectorsA@vectorsB/(abs(np.sqrt(np.sum((vectorsA*vectorsA))))*abs(np.sqrt(np.sum((vectorsB*vectorsB)))))
    coss=float(np.squeeze(coss,axis=1))
    angles=np.arccos(coss)*180/3.14
    cost = vectortA @ vectortB / (abs(np.sqrt(np.sum((vectortA * vectortA)))) * abs(np.sqrt(np.sum((vectortB * vectortB)))))
    cost = float(np.squeeze(cost, axis=1))
    anglet= np.arccos(cost) * 180 / 3.14
    # ----------------------------------------------------------------------------------
    # 只使用阈值判断
    # ----------------------------------------------------------------------------------
    # print(index)
    # print(subA,subB,abs(angles-anglet))
    # 有两个超了就不认为是对齐的
    if subA<0.01 and subB<0.01 and  abs(angles-anglet)<1:
        return True
    else:
        return False
def isMatchB(index,engineeringlpsn,engineeringlptn,*args):
    """
           纯计算，仅采用距离角度的差值
           ------------------------------------------------------------------------------------
           index: 预判断是否对齐点的索引
           engineeringlpsn: 源点云的n对点
           engineeringlptn: 目标点云的n对点
           rgs: 默认为对齐的点的索引
           return: 返回是否符合阈值要求
           """
    matchedA = args[0][0]
    matchedB = args[0][1]
    matchedsA = np.array(engineeringlpsn[matchedA])
    matchedtA = np.array(engineeringlptn[matchedA])
    matchedsB = np.array(engineeringlpsn[matchedB])
    matchedtB = np.array(engineeringlptn[matchedB])
    s = np.array(engineeringlpsn[index])
    t = np.array(engineeringlptn[index])
    vectorsA = np.array(s - matchedsA).reshape([1, 3])
    vectorsB = np.array(s - matchedsB).reshape([3, 1])
    vectortA = np.array(t - matchedtA).reshape([1, 3])
    vectortB = np.array(t - matchedtB).reshape([3, 1])
    subA = abs(abs(np.sqrt(np.sum(vectorsA * vectorsA))) - abs(np.sqrt(np.sum(vectortA * vectortA))))
    subB = abs(abs(np.sqrt(np.sum((vectorsB * vectorsB)))) - abs(np.sqrt(np.sum(vectortB * vectortB))))
    coss = vectorsA @ vectorsB / (
            abs(np.sqrt(np.sum((vectorsA * vectorsA)))) * abs(np.sqrt(np.sum((vectorsB * vectorsB)))))
    coss = float(np.squeeze(coss, axis=1))
    angles = np.arccos(coss) * 180 / 3.14
    cost = vectortA @ vectortB / (
            abs(np.sqrt(np.sum((vectortA * vectortA)))) * abs(np.sqrt(np.sum((vectortB * vectortB)))))
    cost = float(np.squeeze(cost, axis=1))
    anglet = np.arccos(cost) * 180 / 3.14
    # ----------------------------------------------------------------------------------
    # 神经网络+计算，阈值放宽松
    # ----------------------------------------------------------------------------------
    if predict(np.array([[subA, subB, abs(angles - anglet)]])):
        state1 = True
    else:
        state1 = False
    return state1
def isMatchC(index,engineeringlpsn,engineeringlptn,*args):
    """
    神经网络+计算，采用距离角度的比值
    ------------------------------------------------------------------------------------
    index: 预判断是否对齐点的索引
    engineeringlpsn: 源点云的n对点
    engineeringlptn: 目标点云的n对点
    rgs: 默认为对齐的点的索引
    return: 返回是否符合阈值要求
    """
    matchedA=args[0][0]
    matchedB=args[0][1]
    matchedC=args[0][2]
    matchedsA=np.array(engineeringlpsn[matchedA])
    matchedtA=np.array(engineeringlptn[matchedA])
    matchedsB=np.array(engineeringlpsn[matchedB])
    matchedtB=np.array(engineeringlptn[matchedB])
    matchedsC=np.array(engineeringlpsn[matchedC])
    matchedtC = np.array(engineeringlptn[matchedC])
    s=np.array(engineeringlpsn[index])
    t=np.array(engineeringlptn[index])
    vectorsA=np.array(s-matchedsA)
    vectorsB = np.array(s - matchedsB)
    vectorsC = np.array(s - matchedsC)
    vectortA=np.array(t-matchedtA)
    vectortB = np.array(t - matchedtB)
    vectortC = np.array(t - matchedtC)
    subA=abs(np.linalg.norm(vectorsA,ord=2)-np.linalg.norm(vectortA,ord=2))
    subB=abs(np.linalg.norm(vectorsB,ord=2)-np.linalg.norm(vectortB,ord=2))
    subC = abs(np.linalg.norm(vectorsC, ord=2) - np.linalg.norm(vectortC, ord=2))
    cossAB=vectorsA.reshape([1,3])@vectorsB.reshape([3,1])/(np.linalg.norm(vectorsA,ord=2)*np.linalg.norm(vectorsB,ord=2))
    cossAB=float(np.squeeze(cossAB,axis=1))
    anglesAB=np.arccos(cossAB)*180/3.14
    costAB = vectortA.reshape([1,3]) @ vectortB.reshape([3,1]) / (np.linalg.norm(vectortA,ord=2)*np.linalg.norm(vectortB,ord=2))
    costAB = float(np.squeeze(costAB, axis=1))
    angletAB= np.arccos(costAB) * 180 / 3.14
    # ----------------------------------------------------------------------------------
    cossAC = vectorsA.reshape([1,3]) @ vectorsC.reshape([3, 1]) / (np.linalg.norm(vectorsA, ord=2) * np.linalg.norm(vectorsC, ord=2))
    cossAC = float(np.squeeze(cossAC, axis=1))
    anglesAC = np.arccos(cossAC) * 180 / 3.14
    costAC = vectortA.reshape([1,3]) @ vectortC.reshape([3, 1]) / (np.linalg.norm(vectortA, ord=2) * np.linalg.norm(vectortC, ord=2))
    costAC = float(np.squeeze(costAC, axis=1))
    angletAC = np.arccos(costAC) * 180 / 3.14
    # ----------------------------------------------------------------------------------
    cossBC = vectorsB.reshape([1,3]) @ vectorsC.reshape([3, 1]) / (np.linalg.norm(vectorsB, ord=2) * np.linalg.norm(vectorsC, ord=2))
    cossBC = float(np.squeeze(cossBC, axis=1))
    anglesBC = np.arccos(cossBC) * 180 / 3.14
    costBC = vectortB.reshape([1,3]) @ vectortC.reshape([3, 1]) / (np.linalg.norm(vectortB, ord=2) * np.linalg.norm(vectortC, ord=2))
    costBC = float(np.squeeze(costBC, axis=1))
    angletBC = np.arccos(costBC) * 180 / 3.14
    # print(index)
    # print(subA,subB,subC,abs(anglesAB-angletAB),abs(anglesAC-angletAC),abs(anglesBC-angletBC))
    if subA<0.1 and subB<0.1 and subC<0.1 and abs(anglesAB-angletAB)<1.5 and abs(anglesAC-angletAC)<1.5 and abs(anglesBC-angletBC)<1.5:
        return True
    else:
        return False

def A(txtPath):
    """
    使用2条边和1个角
    非批量化处理
    ------------------------------------------------------------------------------------
    param:
        txtPath: superGlue生成的txt
    """

    lps, lpt,_ = superGluetxt2arr(txtPath)
    # ----------------------------------------------------------------------------------
    # 假设两对点是对齐的状态,进行遍历统计投票的情况
    # ----------------------------------------------------------------------------------
    dict = {}
    # 假设两个点是提前对齐的
    length=lps.shape[0]
    ids = [i for i in range(length)]
    # idPairs=[[5,7]]
    idPairs = list(combinations(ids, 2))
    for i in range(length):
        dict[i] = 0
    for idPair in idPairs:
        idPair = np.array(idPair)
        for i in range(length):
            if i not in idPair:
                if isMatchA(i, lps, lpt, idPair):
                    dict[i] += 1
                    dict[idPair[0]] += 1
                    dict[idPair[1]] += 1
    # print("投票结果为:", dict)
    # print("投票数量从小到大的索引为:", sorted(dict.keys(), key=lambda x: dict[x]))
    N = sorted(dict.keys(), key=lambda x: dict[x])
    print("投票结果为:", dict)
    print("投票数量从小到大的索引为:", N)
    return N, np.array(lps)[N[-4:]], np.array(lpt)[N[-4:]]
def B(txtPath):
    """
     使用2条边和1个角
     非批量化处理
     ------------------------------------------------------------------------------------
     param:
         txtPath: superGlue生成的txt
     """

    lps, lpt, _ = superGluetxt2arr(txtPath)
    # ----------------------------------------------------------------------------------
    # 假设两对点是对齐的状态,进行遍历统计投票的情况
    # ----------------------------------------------------------------------------------
    dict = {}
    # 假设两个点是提前对齐的
    length = lps.shape[0]
    ids = [i for i in range(length)]
    # idPairs=[[5,7]]
    idPairs = list(combinations(ids, 2))
    for i in range(length):
        dict[i] = 0
    for idPair in idPairs:
        idPair = np.array(idPair)
        for i in range(length):
            if i not in idPair:
                if isMatchB(i, lps, lpt, idPair):
                    dict[i] += 1
                    dict[idPair[0]] += 1
                    dict[idPair[1]] += 1
    # print("投票结果为:", dict)
    # print("投票数量从小到大的索引为:", sorted(dict.keys(), key=lambda x: dict[x]))
    N = sorted(dict.keys(), key=lambda x: dict[x])
    print("投票结果为:", dict)
    print("投票数量从小到大的索引为:", N)
    return N, np.array(lps)[N[-4:]], np.array(lpt)[N[-4:]]
def C(txtPath):
    """
    使用3条边和3个角
    非批量化处理
    ------------------------------------------------------------------------------------
    param:
        txtPath: superGlue生成的txt
    """
    lps, lpt ,_= superGluetxt2arr(txtPath)
    # ----------------------------------------------------------------------------------
    # 假设两对点是对齐的状态,进行遍历统计投票的情况
    # ----------------------------------------------------------------------------------
    dict = {}
    length = lps.shape[0]
    # 假设两个点是提前对齐的
    ids = [i for i in range(length)]
    # idPairs=[[5,7,9]]
    idPairs = list(combinations(ids, 3))
    for i in range(length):
        dict[i] = 0
    for idPair in idPairs:
        idPair = np.array(idPair)
        for i in range(length):
            if i not in idPair:
                if isMatchC(i, lps, lpt, idPair):
                    dict[i] += 1
                    dict[idPair[0]] += 1
                    dict[idPair[1]] += 1
                    dict[idPair[2]]+=1
    # print("投票结果为:", dict)
    # print("投票数量从小到大的索引为:", sorted(dict.keys(), key=lambda x: dict[x]))
    N = sorted(dict.keys(), key=lambda x: dict[x])
    print("投票结果为:", dict)
    print("投票数量从小到大的索引为:", N)
    return N, np.array(lps)[N[-4:]], np.array(lpt)[N[-4:]]
if __name__=="__main__":
   # path = rootPath = "C:\\Users\\asus\\Desktop\\Hypothesis\\918\\text3d\\38_339.txt"
   # path = "C:\\Users\\asus\\Desktop\\mydata\\test\\output\\190_277.txt"
   # path = "H:\\output_830\\text3d\\366_416.txt"
   # path="H:\\output_830\\text3d\\366_416.txt"
   path = "C:/Users/asus/Desktop/Hypothesis/datasets/190_277.txt"#46_360
   lps,lpt,idDel=superGluetxt2arr(path)
   print(lpt)
   # A(path)
   # B(path)
   # C(path)