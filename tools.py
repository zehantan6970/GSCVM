import numpy as np
import cv2

'''
直接导入此文件，from match2WZH import getpair as gan，使用：lpsn,lptn=gan(str(names[key_point4])+'.png',str(m)+'.png')
'''

def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=1000,fcv=False): #livingroom(depth_sca = 10000)
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
        xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)#(480, 640, 3)
    return xyz

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

#获取像素点的三维坐标
def getpointfrimg(rgb_path,depth_path,depth_cam_matrix,corners,fcv):
    depth = cv2.imread(depth_path, -1)  # cloudbin0_depth
    rgb = cv2.imread(rgb_path, 1)
    # points = depth2xyz(depth, depth_cam_matrix, False, 1000, fcv)
    points = depth2xyz(depth, depth_cam_matrix, False, 1000, fcv)   #10000
    lp = np.squeeze(corners, 1)  # list of point cloud
    lxyz = [] # x,y,z list of point cloud
    for i in range(corners.shape[0]):
        u, v = lp[i][0],lp[i][1]
        bx, by = replacenanpoint(u,v,rgb)
        xyz = points[bx, by]
        if np.isnan(xyz)[0] == True:
            print(rgb_path)
            print("this index{0} is nan!!!".format(i))
            continue
        lxyz.append(xyz)
        #print("x,y= {0},{1} xyz={2}".format(by, bx, xyz))
    return lxyz