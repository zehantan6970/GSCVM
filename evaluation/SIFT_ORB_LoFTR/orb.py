import cv2
import numpy as np
from PIL import Image
import time
def uv2xyz(camera_inter,scalingFactor,depth,uv):
    fx ,fy,centerX,centerY=camera_inter
    # -----------------------------------------------------------------
    # 像素坐标u，v
    # -----------------------------------------------------------------
    u,v=uv

    # -----------------------------------------------------------------
    # 相机坐标X，Y，Z
    # -----------------------------------------------------------------
    Z = depth.getpixel((u, v)) / scalingFactor
    if Z == 0:
        return [0,0,0]
    else:
        X = (u - centerX) * Z / fx
        Y = (v - centerY) * Z / fy
        return [X,Y,Z]

def sift(path_1,path_2,depth_1,depth_2,a):
    img1_gray = cv2.imread(path_1)
    img2_gray = cv2.imread(path_2)

    # 创建ORB特征检测器和描述符
    orb = cv2.ORB_create()
    # 对两幅图像检测特征和描述符
    keypoint1, descriptor1 = orb.detectAndCompute(img1_gray, None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2_gray, None)
    # 获得一个暴力匹配器的对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 利用匹配器 匹配两个描述符的相近成都
    matches = bf.match(descriptor1, descriptor2)
    # 按照相近程度 进行排序
    matches = sorted(matches, key=lambda x: x.distance)
    matches=matches[:20]
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoint1[match.queryIdx].pt
        points2[i, :] = keypoint2[match.trainIdx].pt
    points=zip(points1,points2)
    depth_1=Image.open(depth_1)
    depth_2=Image.open(depth_2)
    for (x1, y1), (x2, y2) in points:
        a.write(str(uv2xyz(CAM,1000,depth_1,(x1,y1))[0])+" "+str(uv2xyz(CAM,1000,depth_1,(x1,y1))[1])+" "+str(uv2xyz(CAM,1000,depth_1,(x1,y1))[2]))
        a.write(" "+str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[0])+" "+str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[1])+" "+str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[2])+"\n")

if __name__=="__main__":
    PATH_1 = "/home/light/gree/Hypothesis/heads_evaluation/color_eval/"
    PATH_1_D = "/home/light/gree/Hypothesis/heads_evaluation/depth_eval/"
    PATH_2 = "/home/light/gree/Hypothesis/heads_evaluation/color_eval/"
    PATH_2_D = "/home/light/gree/Hypothesis/heads_evaluation/depth_eval/"
    TXT_ROOTS = "/home/light/gree/Hypothesis/head_output/text3dorb/"
    PAIRS = "/home/light/gree/Hypothesis/registration_evaluate/pairs.txt"
    CAM = [585, 585, 320, 240]
    with open(PAIRS, mode="r") as r:
       lines=r.readlines()
       start=time.time()
       for l in lines:
           n1=l.split()[0]
           n2=l.split()[1]
           txt = "{}_{}.txt".format(n1.split(".")[0],n2.split(".")[0])
           with open(TXT_ROOTS + txt, mode="a") as a:
               sift(PATH_1+n1, PATH_2+n2, PATH_1_D+n1, PATH_2_D+n2, a)
    end=time.time()
    meantime=(end-start)/len(lines)
    print(meantime)
