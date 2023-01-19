import os
os.chdir("..")
from copy import deepcopy

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
# Load example images
img0_pth = "/home/light/gree/Hypothesis/heads_evaluation/color_eval/15.png"
img1_pth = "/home/light/gree/Hypothesis/heads_evaluation/color_eval/52.png"
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
print(mkpts0)
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
#     # print(d.split()[0][1:-1],d.split()[1][:-2])
#     arr0.append([int(float(d.split()[0][1:-1])), int(float(d.split()[1][:-2]))])
# arr1 = []
# for d in dictmkpts1:
#     # print(d.split()[0][1:-1], d.split()[1][:-2])
#     arr1.append([int(float(d.split()[0][1:-1])), int(float(d.split()[1][:-2]))])
# # Draw
# print(arr0)
np.random.seed(100)
a0=np.random.randint(0,len(mkpts0) ,20)
print(a0)
arr0=mkpts1[a0]
np.random.seed(100)
a1 = np.random.randint(0, len(mkpts1), 20)
arr1 = mkpts1[a1]
color = cm.jet(mconf[:20])
text = [
    'LoFTR',
    'Matches: {}'.format(len(arr0)),
]
make_matching_figure(img0_raw, img1_raw, np.array(arr0), np.array(arr1), color, text=text,path="/home/light/gree/LoFTR-master/result.png")
