from itertools import combinations,combinations_with_replacement
from utils import superGluetxt2arr,isMatch
import numpy as np
import os
if __name__=="__main__":
    rootPath = ".\\test\\output\\"
    txtList = os.listdir(rootPath)
    for t in txtList:
        if t[-3:] == "txt":
            txtPath = rootPath + str(t)
            print("txt的配准结果为:", t)
            lps,lpt = superGluetxt2arr(txtPath)
            dict={}
            id=[0,1,2,3,4,5,6,7,8,9]
            idPairs=list(combinations(id,2))
            for i in range(10):
                dict[i]=0
            for idPair in idPairs:
                idPair=np.array(idPair)
                for i in range(10):
                    if i not in idPair:
                        if isMatch(i,lps,lpt,idPair):
                            dict[i]+=1
                            dict[idPair[0]] += 1
                            dict[idPair[1]] += 1
                        else:
                            dict[i] -= 1
            print("投票结果为:", dict)
            for i in range(10):
                if dict[i]<=15:# 纯计算阈值设为10 神经网络+计算设为15，神经网络设为20
                    dict.pop(i)
            SortIndex=sorted(dict.keys(),key=lambda x:dict[x],reverse=True)
            lst = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
            # 找出票数最高的4对点作为对齐的点
            print(lst[SortIndex[:4]])