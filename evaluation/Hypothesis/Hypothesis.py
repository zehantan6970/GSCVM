from itertools import combinations,combinations_with_replacement
from utils.utils import superGluetxt2arr,isMatchA,isMatchB,isMatchC
import numpy as np
import os
import cv2 as cv
from frozenDir import relativePath
import re

def reID(maskRootPath,idName,lps2D, lpt2D, matchPairs,originImg,reIdPath):
    """
    param:
        maskRootPat: 掩膜文件夹根目录
        idName: pair对的id
        lps2D: source的2维坐标,shape=(n,2)
        lpt2D: target的2维坐标,shape=(n,2)
        matchPairs: 配准的id对,shape=(n,2),n是配准的数量
        reIdPath: 输出reid结果的路径
    """
    # 找到源图像与目标图像的掩膜文件
    imgsList=os.listdir(maskRootPath+idName[0])
    imgtList=os.listdir(maskRootPath+idName[1])
    # 设置列表保存mask的名称
    sourceArr=[[] for i in range(len(lps2D))]
    targetArr = [[] for i in range(len(lpt2D))]
    for matchPair in matchPairs:
        # 找出配准的点在图片上的像素坐标
        xyS=lps2D[matchPair[0]]
        xyT=lpt2D[matchPair[1]]
        # 遍历源图像
        for imgs in imgsList:
            images=cv.imread(maskRootPath+idName[0]+"\\"+imgs)
            areas=len(np.where(images[:,:,0]>0)[0])
            # 像素坐标与numpy相反
            rgbS=images[xyS[1],xyS[0]]
            # 没有分割的区域是黑色为0，有分割的大于0
            if sum(rgbS)>100:
                # 一个点可能落在多个mask上,保存mask的名称
                sourceArr[matchPair[0]].append([imgs,areas])
        for imgt in imgtList:
            imaget=cv.imread(maskRootPath+idName[1]+"\\"+imgt)
            areat = len(np.where(imaget[:, :, 0] > 0)[0])
            rgbS=imaget[xyT[1],xyT[0]]
            if sum(rgbS)>100:
                targetArr[matchPair[0]].append([imgt,areat])
    # ----------------------------------------------------------------------------------
    # 只保留置信度最高的mask
    # ----------------------------------------------------------------------------------
    for i,sonArr in enumerate(sourceArr):
        if len(sonArr):
            sourceArr[i]=[min(sonArr,key=lambda x:x[1])]
    for i, sonArr in enumerate(targetArr):
        if len(sonArr):
            targetArr[i] = [min(sonArr, key=lambda x: x[1])]
    # 初始化一张黑色图像
    img_black=np.zeros((720,2560,3),dtype=np.uint8)
    img_copy=originImg
    # ----------------------------------------------------------------------------------
    # 如果源图像与目标图像的点对齐且对齐的点的区域被成功分割到，将每个reid结果单独保存
    # ----------------------------------------------------------------------------------
    for i,z in enumerate(zip(sourceArr,targetArr)):
        if len(z[0]) and len(z[1]):
            print("reID的结果为",end="")
            z=np.reshape(z,(2,2))
            imgMaskS=cv.imread(maskRootPath + idName[0] + "\\" + z[0][0])
            # confMaxIndex=np.argmax([float(z[0][0].split("_")[0]),float(z[1][0].split("_")[1])])
            # label=z[int(confMaxIndex)][0].split("_")[1]
            imgMaskT = cv.imread(maskRootPath + idName[1] + "\\" + z[1][0])
            img=np.hstack((imgMaskS,imgMaskT))
            img_black=cv.add(img_black,img)
            epch_img=cv.bitwise_and(img_copy,img)
            cof_max_index = np.argmax([float(z[0][0].split("_")[0]), float(z[1][0].split("_")[0])])
            cv.line(epch_img, tuple(lps2D[i]), tuple(lpt2D[i] + np.array([1280, 0])), (0, 255, 0), 2)
            cv.putText(epch_img, z[cof_max_index][0].split("_")[1][:-4], tuple(lps2D[i]), cv.FONT_HERSHEY_COMPLEX, 2.0,
                       (100, 200, 200), 5)
            cv.putText(epch_img, z[cof_max_index][0].split("_")[1][:-4], tuple(lpt2D[i] + np.array([1280, 0])),
                       cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            cv.imwrite(reIdPath+idName[0]+"_"+idName[1]+z[cof_max_index][0].split("_")[1][:-4]+"_"+str(i)+"_.jpg",epch_img)
            print("source中的",z[0][0].split("_")[1][:-4],end="对应")
            print("target中的",z[1][0].split("_")[1][:-4])
    originImg=cv.bitwise_and(originImg,img_black)
    # ----------------------------------------------------------------------------------
    # 如果源图像与目标图像的点对齐且对齐的点的区域被成功分割到，将每个reid结果一起保存在同一个图像中
    # ----------------------------------------------------------------------------------
    for i, z in enumerate(zip(sourceArr, targetArr)):
        if len(z[0]) and len(z[1]):
            z = np.reshape(z, (2, 2))
            cof_max_index=np.argmax([float(z[0][0].split("_")[0]),float(z[1][0].split("_")[0])])
            cv.putText(originImg, z[cof_max_index][0].split("_")[1][:-4], tuple(lps2D[i]), cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            cv.putText(originImg, z[cof_max_index][0].split("_")[1][:-4], tuple(lpt2D[i]+ np.array([1280, 0])), cv.FONT_HERSHEY_COMPLEX, 2.0,(100, 200, 200), 5)
            cv.line(originImg, tuple(lps2D[i]), tuple(lpt2D[i] + np.array([1280, 0])), (0, 255, 0), 2)
    cv.imwrite(reIdPath+idName[0]+"_"+idName[1]+".jpg",originImg)

def batchProcess(maskPath,d3Path, txt2dPath, imgPath, filterImgPath,reIdPath,rgbPath):
    txtList = os.listdir(d3Path)
    for t in txtList:
        if t[-3:] == "txt":
            txtPath = d3Path + str(t)
            # 仅获得有深度的点的三维坐标
            lps, lpt,del_id = superGluetxt2arr(txtPath)
            # ----------------------------------------------------------------
            # output_car需要除以100，918需要乘以10
            # ---------------------------------------------------------------
            lps/=10
            lpt/=10
            # 使用字典对投票结果记录
            dict = {}
            # 生成待固定的点的id
            id = [i for i in range(lps.shape[0])]
            idPairs = list(combinations(id, 2))
            for i in range(lps.shape[0]):
                dict[i] = 0
            for idPair in idPairs:
                idPair = np.array(idPair)
                for i in range(lps.shape[0]):
                    if i not in idPair:
                        if isMatchA(i, lps, lpt, idPair):
                            dict[i] += 1
                            dict[idPair[0]] += 1
                            dict[idPair[1]] += 1
            print("投票结果为:", dict)
            for i in range(lps.shape[0]):
                if dict[i] <= 1:
                    dict.pop(i)
            SortIndex = sorted(dict.keys(), key=lambda x: dict[x], reverse=True)
            print(SortIndex)
            lst = np.array([[i,i] for i in range(lps.shape[0])])
            # 找出票数最高的n对点作为对齐的点
            newMatchPairs=lst[SortIndex[:4]]
            # 仅获得有深度信息的点的二维坐标
            with open(txt2dPath+t) as f:
                xyS=[]
                xyT=[]
                lines=f.readlines()
                for i ,line in enumerate(lines):
                    if i not in del_id:
                        line=list(map(int,line.split(" ")))
                        xyS.append(np.array([line[0],line[1]]))
                        xyT.append(np.array([line[2], line[3]]))
            # 获取图像的名称
            idName = t[:-4].split("_")
            imgS=cv.imread(rgbPath+str(idName[0])+".png")
            imgT=cv.imread(rgbPath+str(idName[1])+".png")
            # 拼接原始图像
            inputs = np.hstack((imgS, imgT))
            reIdInput=np.array(inputs)
            # 在图像上画出正确配准的连线
            for n in newMatchPairs:
                cv.line(inputs,tuple(xyS[n[0]]),tuple(xyT[n[1]]+np.array([640,0])),(0,255,0),2)
            # 将正确配准的图像保存为jpg
            cv.imwrite(filterImgPath+str(t[:-3])+"jpg",inputs)
            # 进行reid操作
            mask_path=maskPath + str(t[:-4]) + "\\"
            # reID(mask_path, idName, xyS, xyT, newMatchPairs,reIdInput,reIdPath)
if __name__=="__main__":
    # 总文件夹命名
    folder="918"
    # superglue生成的配准点对的三维坐标
    d3Path = relativePath()+"/datasets/"+folder+"/text3d/"
    # superglue生成的对齐图像的路径
    imgPath=relativePath()+"/datasets/"+folder+"/img/"
    # 对superglue使用自己的算法过滤后保存对齐图像的路径
    filterImgPath=relativePath()+"/datasets/"+folder+"/filterImg/"
    # superglue生成的配准点对的二维坐标
    text2dPath=relativePath()+"/datasets/"+folder+"/text2d/"
    # mmdection分割的掩膜文件路径
    mask_rootpath = relativePath()+"/datasets/"+folder+ "/mask/"
    # 保存重识别结果的路径
    reid_result= relativePath()+"/datasets/"+folder+ "/reid_result/"
    rgbPath=relativePath()+"/datasets/"+folder+"/rgb/"
    # 进行批量化处理
    batchProcess(mask_rootpath, d3Path, text2dPath, imgPath, filterImgPath, reid_result,rgbPath)