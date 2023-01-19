import cv2
import numpy as np
from PIL import Image
import os
os.chdir("..")
from copy import deepcopy
import time
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
from matplotlib import pyplot as plt
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
_default_cfg = deepcopy(default_cfg)
_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("/home/light/gree/LoFTR-master/weights/indoor_ds_new.ckpt")['state_dict'])
matcher = matcher.eval().cuda()
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
def loftr(path_1,path_2,depth_1,depth_2,a):
    # Load example images
    img0_pth = path_1
    img1_pth = path_2
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    # dictmkpts0 = {}
    # for x, y in zip(mkpts0, mconf):
    #     dictmkpts0[str(x)] = y
    # dictmkpts1 = {}
    # for x, y in zip(mkpts1, mconf):
    #     dictmkpts1[str(x)] = y
    # dictmkpts0 = sorted(dictmkpts0.keys(), key=lambda x: dictmkpts0[x])
    # dictmkpts1 = sorted(dictmkpts1.keys(), key=lambda x: dictmkpts1[x])
    # dictmkpts0 = dictmkpts0[-20:]
    # dictmkpts1 = dictmkpts1[-20:]
    # arr0 = []
    # for d in dictmkpts0:
    #     print(d.split()[0][1:-1],d.split()[1][:-2])
    #     arr0.append([int(float(d.split()[0][1:-1])), int(float(d.split()[1][:-2]))])
    # arr1 = []
    # for d in dictmkpts1:
    #     print(d)
    #     print(d.split()[0][1:-1], d.split()[1][:-2])
    #     arr1.append([int(float(d.split()[0][1:-1])), int(float(d.split()[1][:-2]))])
    np.random.seed(100)
    a0=np.random.randint(0,len(mkpts0) ,20)
    arr0=mkpts1[a0]
    np.random.seed(100)
    a1 = np.random.randint(0, len(mkpts1), 20)
    arr1 = mkpts1[a1]
    points = zip(arr0, arr1)
    depth_1 = Image.open(depth_1)
    depth_2 = Image.open(depth_2)
    for (x1, y1), (x2, y2) in points:
        a.write(str(uv2xyz(CAM, 1000, depth_1, (x1, y1))[0]) + " " + str(
            uv2xyz(CAM, 1000, depth_1, (x1, y1))[1]) + " " + str(uv2xyz(CAM, 1000, depth_1, (x1, y1))[2]))
        a.write(" " + str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[0]) + " " + str(
            uv2xyz(CAM, 1000, depth_2, (x2, y2))[1]) + " " + str(uv2xyz(CAM, 1000, depth_2, (x2, y2))[2]) + "\n")

if __name__=="__main__":
    PATH_1 = "/home/light/gree/Hypothesis/heads_evaluation/color_eval/"
    PATH_1_D = "/home/light/gree/Hypothesis/heads_evaluation/depth_eval/"
    PATH_2 = "/home/light/gree/Hypothesis/heads_evaluation/color_eval/"
    PATH_2_D = "/home/light/gree/Hypothesis/heads_evaluation/depth_eval/"
    TXT_ROOTS = "/home/light/gree/Hypothesis/head_output/text3dloftr/"
    PAIRS = "/home/light/gree/Hypothesis/registration_evaluate/pairs.txt"
    CAM = [585, 585, 320, 240]
    with open(PAIRS, mode="r") as r:
       lines=r.readlines()
       start=time.time()
       for l in lines:
           n1=l.split()[0]
           n2=l.split()[1]
           txt = "{}_{}.txt".format(n1.split(".")[0],n2.split(".")[0])
           with open(TXT_ROOTS + txt, mode="w") as a:
               a.close()
           with open(TXT_ROOTS + txt, mode="a") as a:
               loftr(PATH_1+n1, PATH_2+n2, PATH_1_D+n1, PATH_2_D+n2, a)
    end=time.time()
    meantime=(end-start)/len(lines)
    print(meantime)
