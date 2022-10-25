from FilterNet import Net
import os
import torch
import numpy as np
import torch.nn as nn
from net.FilterNet import Net,mlpNet
import open3d as o3d
from itertools import combinations,combinations_with_replacement
from numpy import *
from match2WZH_TEST import getpair as gan
from pcd import product_pcd
from rotateQuaternion import pos_matrix_to_quat_hm
import cv2

#time_start = time.time()
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())

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
            # output = np.append(output, lpsn[i])
    # output:dtype=list
    return output
def superGluetxt2arr(path):
    """
    param:
        path:存储点云n个三维坐标点的txt
    return:
        lpsn: 转换为数组存储，lpsn=[[x,y,z],...]
        lptn: 转换为数组存储，lptn=[[x,y,z],...]
    """
    with open(path,mode="r") as f:
        lines= f.readlines()
        lpsn=[]
        lptn=[]
        for i, line in enumerate(lines):
            line = line.split(" ")
            lst=np.array(list(map(float,line)))
            lpsn.append(lst[:3])
            lptn.append(lst[3:])

    return lpsn, lptn

def superGluetxt2arr_B(path):
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
            if 0 in lst or lst[3:6] in lptnRepeat or lst[:3] in lpsnRepeat:
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

# #根据CDF进行匹配
# def matchBYbhd(fdhs, fdht):
#     match_matrix = []
#     for i in range(len(fdhs)):
#         count_mat = []
#         for j in range(len(fdht)):
#             count_mat.append(get_Bhadist(fdhs[i], fdht[j]))
#         #print(count_mat)
#         if max(count_mat)<0.3:
#             index = math.inf
#         else:
#             index = count_mat.index(max(count_mat))
#         match_matrix.append([i, index])
#     return match_matrix
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
    net = mlpNet(3, 512, 256, 128, 2) # 输入节点6个，输出节点2个
    net.load_state_dict(torch.load('/home/wzh/supergan/net/weights/mlpA20000.pth'))
    pre = net(features)
    softmax_2 = nn.Softmax(dim=1)
    pre = softmax_2(pre)
    prelabels = pre.argmax(dim=1)
    return prelabels

def isMatch(index,engineeringlpsn,engineeringlptn,*args):
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
    divA=abs(abs(np.sqrt(np.sum(vectorsA*vectorsA)))-abs(np.sqrt(np.sum(vectortA*vectortA))))
    divB = abs(abs(np.sqrt(np.sum((vectorsB* vectorsB)))) -abs(np.sqrt(np.sum(vectortB * vectortB))))
    coss=vectorsA@vectorsB/(abs(np.sqrt(np.sum((vectorsA*vectorsA))))*abs(np.sqrt(np.sum((vectorsB*vectorsB)))))
    coss=float(np.squeeze(coss,axis=1))
    angles=np.arccos(coss)*180/3.14
    cost = vectortA @ vectortB / (abs(np.sqrt(np.sum((vectortA * vectortA)))) * abs(np.sqrt(np.sum((vectortB * vectortB)))))
    cost = float(np.squeeze(cost, axis=1))
    anglet= np.arccos(cost) * 180 / 3.14

    if divA<0.08 and divB<0.08 and  abs(angles-anglet)<0.5:# 0.95 1.05
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

def OnePointMatched(index,engineeringlpsn,engineeringlptn,matchedindex):
    matchedsA = np.array(engineeringlpsn[matchedindex])
    matchedtA = np.array(engineeringlptn[matchedindex])
    s= np.array(engineeringlpsn[index])
    t = np.array(engineeringlptn[index])
    vectors=s-matchedsA
    vectort=t-matchedtA
    vectorsLen=np.linalg.norm(vectors,ord=2)
    vectortLen = np.linalg.norm(vectort,ord=2)
    # print(index)
    # print(vectorsLen/vectortLen)
    if 0.98<vectorsLen/vectortLen<1.02:
        return True
    else:
        return False
def A(txtPath):
    lps, lpt = superGluetxt2arr(txtPath)
    dict = {}
    # idPairs=[[4,16]]
    ids = [i for i in range(len(lps))]
    idPairs = list(combinations(ids, 2))
    # print(len(idPairs))
    for i in range(len(lps)):
        dict[i] = 0
    for idPair in idPairs:
        idPair = np.array(idPair)
        for i in range(len(lps)):
            if i not in idPair:
                if isMatch(i, lps, lpt, idPair):
                    dict[i] += 1
                #     dict[idPair[0]] += 1
                #     dict[idPair[1]] += 1
                # else:
                #     dict[i] -= 1
    N=sorted(dict.keys(), key=lambda x: dict[x])
    print("Voting results:", dict)
    print("The index of votes from small to large is:", N)
    return dict,N,np.array(lps)[N[-10:]],np.array(lpt)[N[-10:]]                                                        #改，取多少superglue得到的点数对
    # return N,np.array(lps),np.array(lpt)
def B(txtPath):
    """
     使用2条边和1个角
     非批量化处理
     ------------------------------------------------------------------------------------
     param:
         txtPath: superGlue生成的txt
     """

    lps, lpt, _ = superGluetxt2arr_B(txtPath)
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
    print("Voting results:", dict)
    print("The index of votes from small to large is:", N)
    return dict,N,np.array(lps)[N[-10:]],np.array(lpt)[N[-10:]]

def rigid_transform_3D(A, B):
   assert len(A) == len(B)

   N = A.shape[0]  # total points
   centroid_A = np.mean(A, axis=0)
   centroid_B = np.mean(B, axis=0)

   # centre the points
   AA = A - np.tile(centroid_A, (N, 1))
   BB = B - np.tile(centroid_B, (N, 1))

   H = np.matmul(np.transpose(AA), BB)
   U, S, Vt = np.linalg.svd(H)
   R = np.matmul(Vt.T, U.T)

   # special reflection case
   if np.linalg.det(R) < 0:
       # print("Reflection detected")
       Vt[2, :] *= -1
       R = np.matmul(Vt.T, U.T)

   T = -np.matmul(R, centroid_A) + centroid_B
   RT = np.eye(4)
   RT[:3, :3] = R
   RT[:3, 3] = T
   RT[3, 3] = 1
   return RT
def every_nth(lst, nth):
    return lst[nth - 0::nth]

def getnames(path_d,jump):                                         #得到所取帧的文件名称（type:list）
    this_p_names = []
    this_ply_names=[]
    with open(path_d, 'r') as f:
        for l in f.readlines():
            itemss = l.split()
            this_p_name=itemss[0]
            this_ply_name=itemss[2]
            # print(this_p_name)
            this_p_names.append(this_p_name)
            this_ply_names.append(this_ply_name)
            names=every_nth(this_p_names,jump)                      #通过此处获得tum的图片名称以及对应的ply的名称，因为tum的ply和深度图同名，而associata里是1.300188 rgb/1.300188.png 1.300211 depth/1.300211.png这种形式
            names_ply=every_nth(this_ply_names,jump)
            names.insert(0,this_p_names[0])
            names_ply.insert(0,this_ply_names[0])
    # print(names)
    return names,names_ply                                         #这个程序里当成depth图的名称


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
    if subA<2 and subB<2 and subC<2 and abs(anglesAB-angletAB)<1.5 and abs(anglesAC-angletAC)<1.5 and abs(anglesBC-angletBC)<1.5:
        return True
    else:
        return False
    # if 0.99<subA<1.01 and 0.99<subA<1.01 and 0.99<subA<1.01 and 0.99<abs(anglesAB/angletAB)<1.01 and 0.99<abs(anglesAC/angletAC)<1.01 and 0.99<abs(anglesBC/angletBC)<1.01:
    #     return True
    # else:
    #     return False
    # if subA<0.01 and subB<0.01 and subC<0.01 and 0.97<abs(anglesAB/angletAB)<1.03 and 0.97<abs(anglesAC/angletAC)<1.03 and 0.97<abs(anglesBC/angletBC)<1.03:
    #     return True
    # else:
    #     return False
def C(txtPath):
    """
    使用3条边和3个角
    非批量化处理
    ------------------------------------------------------------------------------------
    param:
        txtPath: superGlue生成的txt
    """
    lps, lpt = superGluetxt2arr(txtPath)
    # ----------------------------------------------------------------------------------
    # 假设两对点是对齐的状态,进行遍历统计投票的情况
    # ----------------------------------------------------------------------------------
    dict = {}
    # 假设两个点是提前对齐的
    ids = [i for i in range(len(lps))]
    idPairs = list(combinations(ids, 3))
    for i in range(len(lps)):
        dict[i] = 0
    for idPair in idPairs:
        idPair = np.array(idPair)
        for i in range(len(lps)):
            if i not in idPair:
                if isMatchC(i, lps, lpt, idPair):
                    dict[i] += 1
                    dict[idPair[0]] += 1
                    dict[idPair[1]] += 1
                    dict[idPair[2]]+=1
    N = sorted(dict.keys(), key=lambda x: dict[x])
    print("Voting results:", dict)
    print("The index of votes from small to large is:", N)
    return dict,N, np.array(lps)[N[-10:]], np.array(lpt)[N[-10:]]

def choice_number_rtpoints(threshold_vote,dict_vote,the_index,lptn, lpsn,rt_all):
    if dict_vote[the_index[-4]] <= threshold_vote:
        lps, lpt = lpsn[-4::1, :], lptn[-4::1, :]
        print('Less votes, use ', lps.shape[0], 'pairs of points')
        rt = rigid_transform_3D(lpt, lps)  # 算当前帧和相邻上一帧的rt
        rt_all.append(rt)
        # print('此应该取4个', lps.shape[0], '个点来计算rt')
    elif dict_vote[the_index[-4]] > threshold_vote and dict_vote[the_index[-10]] < threshold_vote:
        print('cut')
        for a in range(5, 11):  # 5,11
            if dict_vote[the_index[-a]] <= threshold_vote:
                lps = np.delete(lpsn, list(range(0, 11 - a)), axis=0)
                lpt = np.delete(lptn, list(range(0, 11 - a)), axis=0)
                rt = rigid_transform_3D(lpt, lps)  # 算当前帧和相邻上一帧的rt
                rt_all.append(rt)
                break
            elif dict_vote[the_index[-a]] > threshold_vote:
                continue
    else:
        lpt, lps = lptn, lpsn
        rt = rigid_transform_3D(lptn, lpsn)
        rt_all.append(rt)
    print('Pairs participating in the calculation:', lps.shape[0])
    return rt,rt_all


def TransformPointCloud(pcd_T,rt_all,p_rt_all):
    import copy
    p_rt = copy.deepcopy(pcd_T)
    for v in reversed(rt_all):  # 循环遍历当前rt_all里的所有rt
        p_rt = p_rt.transform(v)
    p_rt_all.append(p_rt)
    print('camera coordinate origin​​ to initial:', np.array(p_rt.points)[0])
    return p_rt_all
def WriteRt(rt_all,pathRtxt, w_all):
    w = rt_all[-1]
    if len(rt_all) >= 2:
        for u in reversed(rt_all[:-1]):
            w = np.matmul(w, u)
    # p_rt_all.append(p_rt)                                                         # 将转换后的当前帧进行储存
    w_all.append(w)
    # WriteRttxt = open('./output/LivingAllRt.txt', 'w')                            # 转换rt成7组数,前三个为平移,后四个为
    WriteRttxt = open(pathRtxt, 'w')
    for x in w_all:
        y = pos_matrix_to_quat_hm(x)
        for z in y:
            WriteRttxt.write(str(z) + ' ')
        WriteRttxt.write('\n')
    WriteRttxt.close()
    print('RT have recorded')
def WriteCameraXYZ(p_rt_all, pathCtxt):
    # WriteCameraXYZtxt = open('./output/cameraxyz.txt', 'w')
    WriteCameraXYZtxt = open(pathCtxt, 'w')
    for h in p_rt_all:
        WriteCameraXYZtxt.write(str(np.array(h.points)[0]) + '\n')
    WriteCameraXYZtxt.close()
    print('CameraXYZ have recorded')
    return
def DeleteWrongDepthPoints(path):
    with open(path + 'associate.txt', 'r') as f:
        for l in f.readlines():
            items = l.split()
            rgb = path + items[1]
            depth = path + items[3]
            rgbimg = cv2.imread(rgb)
            depthimg = cv2.imread(depth, -1)
            rgbimg[depthimg == 0] = 0
            cv2.imshow('img', rgbimg)
            cv2.waitKey(1000)
            cv2.imwrite('/home/wzh/supergan/tum-data/datatest/rgb/' + items[0] + '.png', rgbimg)
    cv2.destroyAllWindows()
    print('The dataset has been processed')
    return

def DeleteWrongDepthPoints(path):
    with open(path + 'associate.txt', 'r') as f:
        for l in f.readlines():
            items = l.split()
            rgb = path + items[1]
            depth = path + items[3]
            rgbimg = cv2.imread(rgb)
            depthimg = cv2.imread(depth, -1)
            rgbimg[depthimg == 0] = 0
            cv2.imshow('img', rgbimg)
            cv2.waitKey(1000)
            cv2.imwrite('path'+'/rgb/' + items[0] + '.png', rgbimg)
    cv2.destroyAllWindows()
    print('The dataset has been processed')
    return

def DeleteWrongDepthPointsOne(path,name_s,name_t,names,names_depth):
    rgbs = path + 'rgb/'+name_s+'.png'
    depths = path + 'depth/'+names_depth[names.index(name_s)]+'.png'
    rgbt = path + 'rgb/'+name_t+'.png'
    deptht = path + 'depth/'+names_depth[names.index(name_t)]+'.png'
    rgbimgs = cv2.imread(rgbs)
    depthimgs = cv2.imread(depths, -1)
    rgbimgt = cv2.imread(rgbt)
    depthimgt = cv2.imread(deptht, -1)
    rgbimgs[depthimgs == 0] = 0
    rgbimgt[depthimgt == 0] = 0
    cv2.imshow('img1', rgbimgs)
    cv2.imshow('img2', rgbimgt)
    cv2.waitKey(1000)
    cv2.imwrite('path'+'rgb/' + str(name_s) + '.png', rgbimgs)
    cv2.imwrite('path' + 'rgb/' + str(name_t) + '.png', rgbimgs)
    cv2.destroyAllWindows()
    print('The dataset has been processed')
    return




if __name__=="__main__":
    if not os.path.exists('./output'):
        os.makedirs('./output')
    path_o = '/home/wzh/supergan/tum-data/'                           # 改
    data_path = '/home/wzh/supergan/tum-data/datatest/associate.txt'  # 改
    path_match = '/home/wzh/supergan/tum-data/datatest/'              # 改，datatest是数据集的名称
    names, names_depth = getnames(data_path,jump=1)                   # 改，得到names、depthnames,jump是多少帧检测一次，等于1时不跳帧
    fx, fy, cx, cy = 622.28, -622.89, 642.33, 366.55                  # 改，内参
    rt_all = []  # 储存所有rt
    p_rt_all = []  # 储存所有被rt转换过的点云
    w_all = []
    DeleteWrongDepthPoints(path_match)                                #改,是否对数据集中的图片进行去除深度为零点操作
    for z in range((len(names) - 1)):                                 #改，拼的范围
        # for m ,n in zip(names[(len(names) - 1)],names_depth[(len(names_depth) - 1)]):
        lpsn, lptn = gan(names[z],names_depth[z],names[z+1],names_depth[z+1],fx, fy, cx, cy,path_match,super_point=50)    #super_point是superglue生成多少个点
        path = path_o + 'texttest/' + names[z] + "_" + names[z + 1] + ".txt"  # 漏斗所读取txt路径
        print( names[z] + "_" + names[z + 1])
        pcd_T=product_pcd( path_match+'rgb/'+names[z+1]+'.png',path_match+'depth/'+names_depth[z+1]+'.png',fx, fy, cx, cy)
        dict_vote, the_index, lpsn, lptn = A(path)                                                                           # 改，用哪个函数
        rt, rt_all = choice_number_rtpoints(100,dict_vote,the_index,lptn,lpsn,rt_all)                      # 改,筛点的阈值
        p_rt_all = TransformPointCloud(pcd_T,rt_all,p_rt_all)
        print('-' * 150)
    WriteRt(rt_all, './output/TumAllRt.txt',w_all)  # 改,是否写入rt转七元数
    WriteCameraXYZ(p_rt_all, './output/TumCameraxyz.txt')  # 改,是都记录相机坐标原点变化
    final =product_pcd( path_match+'rgb/'+names[0]+'.png',path_match+'depth/'+names_depth[0]+'.png',fx, fy, cx, cy)    #生成点云rgbpicturepath,depthpicturepath,fx, fy, cx, cy):
    for h in range(len(p_rt_all)):  # 把第一帧分别拼接之前算出的像第一帧转化的点云
        final += p_rt_all[h]
    # final = final.voxel_down_sample(voxel_size=0.01)                #改，体素下采样
    # final = final.remove_statistical_outlier(20, 0.5)               #改，基于统计的方式剔除点云中离群点
    # pcd2 = final[0]                                                 #滤波返回点云，和点云索引
    # final=final.remove_statistical_outliers(nb_neighbors=16, std_ratio=2.0)
    o3d.io.write_point_cloud("tumtest10.112.ply", final, write_ascii=True)   #gai储存文件名
    o3d.visualization.draw_geometries([final],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
