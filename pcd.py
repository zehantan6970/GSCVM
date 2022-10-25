import cv2
import numpy as np
import open3d as o3d

def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=1000,fcv=False):
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

def rgb2xyz(rgbmap, depth_cam_matrix, flatten=False, depth_scale=1000,fcv=False):
    #https://blog.csdn.net/qq_24815615/article/details/113276627
    h, w = rgbmap.shape[0], rgbmap.shape[1]
    xyzs = []
    rgbs = []
    for row in range(h):
        for col in range(w):
            r, g, b = rgbmap[row, col][2], rgbmap[row, col][1], rgbmap[row, col][0]
            rgb = [r/255,g/255,b/255]
            # rgb = np.dstack((r, g, b)) if flatten == False else np.dstack((r, g, b)).reshape(-1, 3)
            rgbs.append(rgb)
    return rgbs
def product_pcd(rgbpicturepath,depthpicturepath,fx, fy, cx, cy):
    depth_cam_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    masked = cv2.imread(depthpicturepath,-1)
    getmaskimg = cv2.imread(rgbpicturepath,-1)
    points = depth2xyz(masked, depth_cam_matrix, True, 1000, fcv=False)   #masked：深度图
    pc = o3d.geometry.PointCloud()
    xyzs = points.copy()
    xyzs = xyzs.reshape(-1, 3)
    #print(xyzs)
    pc.points = o3d.utility.Vector3dVector(xyzs)
    #o3d.visualization.draw_geometries([pc], window_name="mypoints", width=800, height=600)
    colors = rgb2xyz(getmaskimg, depth_cam_matrix, True, 1000, fcv=False)    #getmasking：彩色图
    pc.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pc],
    #                                   window_name="pcd",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)
    return pc
if __name__ =='__main__':
    print()