from utils.utils import A
import numpy as np
import os
from itertools import combinations
def isMatchA(index,engineeringlpsn,engineeringlptn,label,w,*args):
    """
    纯计算，仅采用距离角度的差值
    ------------------------------------------------------------------------------------
    index: 预判断是否对齐点的索引
    engineeringlpsn: 源点云的n对点
    engineeringlptn: 目标点云的n对点
    args: 默认为对齐的点的索引
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
    # if label[matchedA]==0 or label[matchedB]==0 or label[index]==0:
    #     # print(subA,subB,abs(angles-anglet),0)
    #     w.write(str(subA) + " " + str(subB) + " " + str(abs(angles - anglet)) + " 0\n")
    # else:
    #     # print(subA, subB, abs(angles - anglet), 1)
    #     w.write(str(subA)+" "+str(subB)+" "+str(abs(angles - anglet))+" 1\n")
    # 有两个超了就不认为是对齐的
    if subA<0.08 and subB<0.08 and  abs(angles-anglet)<1:
        w.write(str(subA) + " " + str(subB) + " " + str(abs(angles - anglet)) + " 1\n")
        return True
    else:
        w.write(str(subA) + " " + str(subB) + " " + str(abs(angles - anglet)) + " 0\n")
        return False
def readTxt(path):
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
        label=[]
        for i, line in enumerate(lines):
            line = line.strip().split(" ")
            lst = list(map(float, line))
            if lst[:3] in lpsnArr and lst[:3] not in lpsnRepeat:
                lpsnRepeat.append(lst[:3])
            elif lst[3:6] in lptnArr and lst[3:6] not in lptnRepeat:
                lptnRepeat.append(lst[3:6])
            else:
                lpsnArr.append(lst[:3])
                lptnArr.append(lst[3:6])
        for i, line in enumerate(lines):
            line = line.strip().split(" ")
            lst=list(map(float,line))
            if 0 in lst[:6] or lst[3:6] in lptnRepeat or lst[:3] in lpsnRepeat:
                del_id.append(i)
            else:
                lpsn.append(lst[:3])
                lptn.append(lst[3:6])
                label.append(lst[6])
    return np.array(lpsn),np.array(lptn),np.array(del_id),np.array(label)
def generateA(lps,lpt,label,w):
    """
        生成训练集
        ------------------------------------------------------------------------------------
        param:
            lps: 源点云的x,y,z
            lpt: 目标点云的x,y,z
            label: 是否对齐，对齐为1，否则为0
            w: open()
        return:
            N: 投票数量从小到大的索引
            outLps: 投票数量前n的lps坐标
            outLpt: 对应的前n的lpt坐标
        """

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
                if isMatchA(i, lps, lpt, label,w,idPair):
                    dict[i] += 1
                    dict[idPair[0]] += 1
                    dict[idPair[1]] += 1
    # print("投票结果为:", dict)
    # print("投票数量从小到大的索引为:", sorted(dict.keys(), key=lambda x: dict[x]))
    N = sorted(dict.keys(), key=lambda x: dict[x])
    print("投票结果为:", dict)
    print("投票数量从小到大的索引为:", N)
    outLps=np.array(lps)[N[-4:]]
    outLpt=np.array(lpt)[N[-4:]]
    return N, outLps, outLpt
if __name__=="__main__":
    txtNames=os.listdir("/net/originData")
    txtNames=sorted(txtNames,key=lambda x:int(x.split("_")[0]))
    rootTxtPath="/home/light/gree/align/Hypothesis/net/originData/"
    with open("trainData.txt",mode="w") as w:
        for txt in txtNames:
            txtPath=rootTxtPath+txt
            lps,lpt,delID,label=readTxt(txtPath)
            generateA(lps,lpt,label,w)
