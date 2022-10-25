import numpy as np
import math
from FilterNet import Net
import torch
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import scipy.spatial.distance as distance
import open3d as o3d
import copy
import time
import matplotlib as plt
import matplotlib.cm
from itertools import combinations,combinations_with_replacement
from numpy import *
from match2WZH_living import getpair as gan
from pcd import product_pcd

#time_start = time.time()

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
    features = torch.tensor(features, dtype=torch.float32)
    net = Net(3, 512, 256, 128, 2)  # 输入节点6个，输出节点2个
    net.load_state_dict(torch.load('./pth/filter20000.pth'))
    pre = net(features)
    softmax_2 = nn.Softmax(dim=0)
    pre = softmax_2(pre)
    prelabels = pre.argmax(dim=0)
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

    if divA<0.08 and divB<0.08 and  abs(angles-anglet)<0.4:# 0.95 1.05
        return True
    else:
        return False

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
    print("投票结果为:", dict)
    print("投票数量从小到大的索引为:", N)
    return dict,N,np.array(lps)[N[-10:]],np.array(lpt)[N[-10:]]              # 改，得到superglue投票的前10对
    # return N,np.array(lps),np.array(lpt)
def B(txtPath):
    lps, lpt = superGluetxt2arr(txtPath)
    ids=[i for i in range(10)]
    dict = {}
    for i in range(10):
        dict[i]=0
    for id in ids:
        for i in range(10):
            if i != id:
                if OnePointMatched(i,lps,lpt,id):
                    dict[i]+=1
                #     dict[id]+=1
                # else:
                #     dict[i]-=1
    M=sorted(dict.keys(),key=lambda x:dict[x])
    print('M的数据格式是',type(M))
    print("投票结果为:",dict)
    print("投票数量从小到大的索引为:",M)
    return M
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

def getnames_living(path_d,jump):                                            #得到所取帧的文件名称（type:list）
    this_p_names = []
    with open(path_d, 'r') as f:
        for l in f.readlines():
            itemss = l.split()
            this_p_name=itemss[0]
            # print(this_p_name)
            this_p_names.append(this_p_name)
            names=every_nth(this_p_names,jump)
            names.insert(0,this_p_names[0])
    # print(names)
    return names



def livingroom_ply_to_rgb(livingroom_path):
    with open(livingroom_path, 'r') as f:
        lines = f.readlines()
        ply_n = []
        rgb_n = []
        a = len(lines)
        for i in range(0, len(lines), 2):
            # line = lines[i].strip("\n")
            # line = lines[i].split(" ")
            ply_n.append(int(lines[i].strip("\n").replace(' ', '')))

            rgb_n.append(int(lines[i + 1].strip("\n").replace(' ', '')))
        return ply_n,rgb_n

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
    if subA<0.01 and subB<0.01 and subC<0.01 and abs(anglesAB-angletAB)<1 and abs(anglesAC-angletAC)<1 and abs(anglesBC-angletBC)<1:
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
    print("投票结果为:", dict)
    print("投票数量从小到大的索引为:", N)
    return dict,N, np.array(lps)[N[-10:]], np.array(lpt)[N[-10:]]

if __name__ == "__main__":
    path_o = '/home/wzh/supergan/tum-data/'                                     #改，文件夹tum-data路径
    data_path = '/home/wzh/supergan/tum-data/living_room_png/associations.txt'  #改，livingroom的associations.txt的路径
    path_match = '/home/wzh/supergan/tum-data/living_room_png/'                 #改，living_room_png的路径+‘/’
    names= getnames_living(data_path,jump=5)                                    #改，得到names、plynames，jump=1时不跳帧，jump=5时，50-55,55-60....
    ply_nlist, rgb_nlist = livingroom_ply_to_rgb('/home/wzh/supergan/tum-data/living_room_png/ply_to_png')  #改得到livingroom点云文件名称列表以及rgb文件名称列表
    fx, fy, cx, cy = 481.20, -480.0, 319.50, 239.50                             # livingname的内参
    rt_all = []                                                                 # 储存所有rt
    p_rt_all = []                                                               # 储存所有被rt转换过的点云
    for z in range(names.index('50')+1,len(names)-1):                           #这里从第50帧开始拼
        a_pcd = rgb_nlist.index(int(names[z]))
        a_pct = rgb_nlist.index(int(names[z + 1]))
        ply_pcd_name = ply_nlist[a_pcd]
        ply_pct_name = ply_nlist[a_pct]
        path_ply1 = path_o + "living_room_png/ply/" + str(ply_pcd_name) + ".ply"
        path_ply2 = path_o + "living_room_png/ply/" + str(ply_pct_name) + ".ply"
        pcd = o3d.io.read_point_cloud(path_ply1)
        pcd_T = o3d.io.read_point_cloud(path_ply2)                              # 找到当前帧得到的对应文件
        lpsn, lptn = gan(names[z], names[z], names[z + 1], names[z + 1], fx, fy, cx, cy,path_match,super_point=40)  # 改，super_point值得:
        path = path_o + 'textliving/' + names[z] + "_" + names[z + 1] + ".txt"  # 漏斗所读取txt路径
        print(names[z] + "_" + names[z + 1])
        dict_vote, the_index, lps, lpt = A(path)                                #改，使用哪种函数
        threshold_vote=500                                                     ##改，一个修改计算rt时点对数的阈值
        if dict_vote[the_index[-4]] <= threshold_vote:
            lps, lpt = lpsn[-4::1, :], lptn[-4::1, :]
            print('此应该取6个', lps.shape[0], '个点来计算rt')
            rt = rigid_transform_3D(lpt, lps)  # 算当前帧和相邻上一帧的rt
            rt_all.append(rt)
            # print('此应该取4个', lps.shape[0], '个点来计算rt')
        elif dict_vote[the_index[-4]] > threshold_vote and dict_vote[the_index[-10]]<threshold_vote:
            print('进行裁剪')
            for a in range(5, 11):   #5,11
                if dict_vote[the_index[-a]] <= threshold_vote:
                    lps = np.delete(lps, list(range(0, 11 - a)), axis=0)
                    lpt = np.delete(lpt, list(range(0, 11 - a)), axis=0)
                    rt = rigid_transform_3D(lpt, lps)                            # 算当前帧和相邻上一帧的rt
                    rt_all.append(rt)
                    break
                elif dict_vote[the_index[-a]] > threshold_vote:
                    continue
        else:
            rt = rigid_transform_3D(lpt, lps)
            rt_all.append(rt)
        print('此时取',lps.shape[0],'个点来计算rt')
        # rt = rigid_transform_3D(lptn, lpsn)                                    #直接用superglue的点
        # rt_all.append(rt)
        import copy
        p_rt = copy.deepcopy(pcd_T)                                               # 对当前帧进行拷贝
        w=rt_all[-1]
        for v in reversed(rt_all):                                                # 循环遍历当前rt_all里的所有rt
            if len(rt_all)>=2:
                w=np.matmul(w,v+1)

            p_rt = p_rt.transform(v)                                              # 对当前帧进行转换（倒序）
        p_rt_all.append(p_rt)                                                     # 将转换后的当前帧进行储存
        w_all.append(w)
    for




    np.matmul(A, B)





    final = o3d.io.read_point_cloud(path_o + "living_room_png/ply/" + "167" + ".ply")  #  改，起始帧
    for h in range(len(p_rt_all)):                                                # 把起始帧分别拼接之前算出的像第一帧转化的点云
        final += p_rt_all[h]
    o3d.io.write_point_cloud("v-livingroom.new.ply", final, write_ascii=True)       #改，储存点云并命名
    o3d.visualization.draw_geometries([final],                                      #可视化
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                  mesh_show_back_face=False)

