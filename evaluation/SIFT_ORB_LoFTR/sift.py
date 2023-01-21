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
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    # h1, w1 = img1_gray.shape[:2]
    # h2, w2 = img2_gray.shape[:2]

    # vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    # vis[:h1, :w1] = img1_gray
    # vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2])
    # post1 = np.int32([kp1[pp].pt for pp in p1])
    # post2 = np.int32([kp2[pp].pt for pp in p2])+np.array([w1,0])

    # for (x1, y1), (x2, y2) in zip(post1, post2):
    #     cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
    # cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    # cv2.imshow("match", vis)
    # cv2.waitKey(0)
    # post2-=np.array([w1,0])
    return post1,post2

def sift(path_1,path_2,depth_1,depth_2,a):
    img1_gray = cv2.imread(path_1)
    img2_gray = cv2.imread(path_2)

    sift = cv2.SIFT_create().create()
    # sift = cv2.SURF()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # # BFmatcher with default parms
    # bf = cv2.BFMatcher(cv2.NORM_L2)
    # matches = bf.knnMatch(des1, des2, k=2)
    # 使用flann匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 设置阈值，越高精度越高，匹配的数量越少
    matches = flann.knnMatch(des1, des2, k=2)
    matches=sorted(matches,key=lambda x:abs(x[0].distance-x[1].distance),reverse=False)
    goodMatch = []
    for m, n in matches:
        if m.distance <= 0.8* n.distance:
            goodMatch.append(m)
    np.random.seed(100)
    a0 = np.random.randint(0, len(goodMatch), 20)
    points1,points2=drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch)
    depth_1=Image.open(depth_1)
    depth_2=Image.open(depth_2)
    for (x1, y1), (x2, y2) in zip(points1[a0],points2[a0]):
        a.write(str(uv2xyz(CAM,1000,depth_1,(x1,y1))[0])+" "+str(uv2xyz(CAM,1000,depth_1,(x1,y1))[1])+" "+str(uv2xyz(CAM,1000,depth_1,(x1,y1))[2]))
        a.write(" "+str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[0])+" "+str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[1])+" "+str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[2])+"\n")
PATH_1="/home/light/gree/Hypothesis/heads_evaluation/color_eval/"
PATH_1_D="/home/light/gree/Hypothesis/heads_evaluation/depth_eval/"
PATH_2="/home/light/gree/Hypothesis/heads_evaluation/color_eval/"
PATH_2_D="/home/light/gree/Hypothesis/heads_evaluation/depth_eval/"
txt_root="/home/light/gree/Hypothesis/head_output/text3dsift/"
if __name__=="__main__":
       pairs="/home/light/gree/Hypothesis/registration_evaluate/pairs.txt"
       CAM = [585, 585, 320, 240]
       with open(pairs,mode="r") as r:
           start = time.time()
           lines=r.readlines()
           for l in lines:
               n1=l.split()[0]
               n2=l.split()[1]
               txt = "{}_{}.txt".format(n1.split(".")[0],n2.split(".")[0])
               with open(txt_root+txt, mode="w") as a:
                   a.close()
               with open(txt_root+txt, mode="a") as a:
                   sift(PATH_1+n1, PATH_2+n2, PATH_1_D+n1, PATH_2_D+n2, a)
       end = time.time()
       meantime = (end - start) / len(lines)
       print(meantime)

