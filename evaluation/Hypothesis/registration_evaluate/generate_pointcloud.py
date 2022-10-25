# coding=utf-8
import numpy as np
from argparse import Namespace
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import math
import os
from frozenDir import relativePath
fx, fy, cx, cy = 481.20, -480.0, 319.50, 239.50  # Robust indoorhttps://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
# fx, fy, cx, cy = 573.702,574.764,324.702,240.97 #scene0002_00
# fx, fy, cx, cy =585,585,320,240 #rgbd-frames

camera_matrix = {'xc': cx, 'zc': cy, 'fx': fx,'fy':fy}
camera_matrix = Namespace(**camera_matrix)
def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=10000,fcv=False):
    #https://blog.csdn.net/qq_24815615/article/details/113276627
    if fcv == False:
        fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
        cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
        h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
        z = depth_map / depth_scale
        x = (w - cx) * z / fx
        y = (h - cy) * z / fy
        xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    else:
        xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz
def rgb2xyz(rgbmap, depth_cam_matrix, flatten=False, depth_scale=1000,fcv=False):
    h, w = rgbmap.shape[0], rgbmap.shape[1]
    rgbs = []
    for row in range(h):
        for col in range(w):
            r, g, b = rgbmap[row, col][2], rgbmap[row, col][1], rgbmap[row, col][0]
            rgb = [r/255,g/255,b/255]
            rgbs.append(rgb)
    return rgbs
def replacenanpoint(x,y,img):
    x,y = y,x
    b0, g0, r0 = img[x, y]
    b1, g1, r1 = img[x, y - 1]
    b2, g2, r2 = img[x + 1, y - 1]
    b3, g3, r3 = img[x + 1, y]
    b4, g4, r4 = img[x + 1, y + 1]
    b5, g5, r5 = img[x, y + 1]
    b6, g6, r6 = img[x - 1, y + 1]
    b7, g7, r7 = img[x - 1, y]
    b8, g8, r8 = img[x - 1, y - 1]
    s0,s1,s2,s3,s4 = int(b0)+int(g0)+int(r0),int(b1)+int(g1)+int(r1),int(b2)+int(g2)+int(r2),int(b3)+int(g3)+int(r3),int(b4)+int(g4)+int(r4)
    s5,s6,s7,s8 = int(b5) + int(g5) + int(r5), int(b6) + int(g6) + int(r6), int(b7) + int(g7) + int(r7), int(b8) + int(g8) + int(r8)
    s = [s0,s1,s2,s3,s4,s5,s6,s7,s8]
    maxid = s.index(max(s))
    if maxid == 0:
        return x,y
    if maxid == 1:
        return x, y - 1
    if maxid == 2:
        return x + 1, y - 1
    if maxid == 3:
        return x + 1, y
    if maxid == 4:
        return x + 1, y + 1
    if maxid == 5:
        return x, y + 1
    if maxid == 6:
        return x - 1, y + 1
    if maxid == 7:
        return x - 1, y
    if maxid == 8:
        return x - 1, y - 1
def getpointfrimg(rgb_path,depth_path,depth_cam_matrix,corners,fcv):
    depth = cv2.imread(depth_path, -1)
    rgb = cv2.imread(rgb_path, 1)
    points = depth2xyz(depth, depth_cam_matrix, False, 10000, fcv)
    lp = np.squeeze(corners, 1)
    lxyz = []
    for i in range(corners.shape[0]):
        u, v = lp[i][0],lp[i][1]
        bx, by = replacenanpoint(u,v,rgb)
        xyz = points[bx, by]
        if np.isnan(xyz)[0] == True:
            print("this index{0} is nan!!!".format(i))
            continue
        lxyz.append(xyz)
    return lxyz
# -----------------------------------------------------------------------
# 生成单个点云
# -----------------------------------------------------------------------
def generateSinglePointcloud():
    rgb_path = "/media/light/LIGHT/datasets/living_room_png/rgb/499.png"
    depth_path ="/media/light/LIGHT/datasets/living_room_png/depth/499.png"
    ply_path="499.ply"
    depth = cv2.imread(depth_path, -1)
    rgb = cv2.imread(rgb_path)
    rgb=cv2.resize(rgb,(640,480))
    depth_cam_matrix = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
    fcv = False
    points = depth2xyz(depth, depth_cam_matrix, False,10000,fcv)
    pc = o3d.geometry.PointCloud()
    xyzs = points.copy()
    xyzs = xyzs.reshape(-1,3)
    pc.points = o3d.utility.Vector3dVector(xyzs)
    # o3d.visualization.draw_geometries([pc], window_name="mypoints", width=640, height=480)
    colors = rgb2xyz(rgb,depth_cam_matrix, True,10000,fcv)
    pc.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pc], window_name="myrgbpoints", width=640, height=480)
    o3d.io.write_point_cloud(ply_path , pc)

# -----------------------------------------------------------------------
# 生成多个点云
# -----------------------------------------------------------------------
def generateBatchCloud():
    rgb_root_path = relativePath()+"/registration_evaluate/heads_eval/heads/color/"
    depth_root_path =relativePath()+"/registration_evaluate/heads_eval/heads/depth/"
    ply_root_path=relativePath()+"/registration_evaluate/heads_eval/heads/ply/"
    rgb_names=os.listdir(rgb_root_path)
    for rgb_name in rgb_names:
        rgb_path=rgb_root_path+rgb_name
        depth_path =depth_root_path+rgb_name.split(".")[0]+".depth.png"
        ply_path=ply_root_path+rgb_name.split(".")[0]+".ply"
        depth = cv2.imread(depth_path, -1)
        rgb = cv2.imread(rgb_path)
        rgb=cv2.resize(rgb,(640,480))
        depth_cam_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        fcv = False
        points = depth2xyz(depth, depth_cam_matrix, False,1000,fcv)
        pc = o3d.geometry.PointCloud()
        xyzs = points.copy()
        xyzs = xyzs.reshape(-1,3)
        pc.points = o3d.utility.Vector3dVector(xyzs)
        colors = rgb2xyz(rgb,depth_cam_matrix, True,1000,fcv)
        pc.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pc], window_name="myrgbpoints", width=640, height=480)
        o3d.io.write_point_cloud(ply_path , pc)
# generateBatchCloud()
generateSinglePointcloud()