# coding=utf-8
import numpy as np
from argparse import Namespace
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d


def xyz2depth(points, depth_cam_matrix, h,w, depth_scale=1000):
    for i in range(points.shape[0]):
        xyz = points[i]
        vu = xyz2vu(xyz, depth_cam_matrix)
        depth = xyz[2] * depth_scale
        uv = np.array([[vu[1], vu[0]]])
        try:
            pix_rgb = im.getpixel((uv[0][0], uv[0][1]))
        except Exception as e:
            print("image index out of range", (uv[0][0], uv[0][1]))
        x = int(uv[0][0])
        y = int(uv[0][1])
        depth = int (depth)
        im.putpixel((x, y), depth)
    return im

def xyz2vu(xyz, depth_cam_matrix, depth_dist_coeff=np.zeros(5)):
    # 畸变类型带添加
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    xyz = np.array(xyz).flatten()
    x, y = xyz[0] / xyz[2], xyz[1] / xyz[2]
    r2 = x * x + y * y
    f = 1 + depth_dist_coeff[0] * r2 + depth_dist_coeff[1] * r2 * r2 + depth_dist_coeff[1] * r2 * r2 * r2
    x *= f
    y *= f
    dx = x + 2 * depth_dist_coeff[2] * x * y + depth_dist_coeff[3] * (r2 + 2 * x * x)
    dy = y + 2 * depth_dist_coeff[3] * x * y + depth_dist_coeff[2] * (r2 + 2 * y * y)
    x, y = dx, dy
    u, v = x * fx + cx, y * fy + cy
    vu = np.int0([v, u])
    return vu

fx, fy, cx, cy = 481.20, -480.0, 319.50, 239.50  # Robust indoorhttps://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
# fx, fy, cx, cy = 475.9, 475.9, 315.5, 245.4  # intel sr300
# fx, fy, cx, cy = 367.28, 367.28, 255.16, 211.82 #kinectv2
# fx, fy, cx, cy = 589.3, 589.8, 321.14, 235.56  # kinectv1
depth_cam_matrix = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
w,h = 640,480
im = Image.new('I', (w, h))
temp = o3d.io.read_point_cloud("F:\Gree\Data\living_room_eval\livingroom2-fragments-ply/cloud_bin_0.ply")#camera121dpt_rgb425-512-012701.ply  camera222dpt_rgb425-512-012701.ply
o3d.visualization.draw_geometries([temp], window_name="oldpoints", width=680, height=480)
# filter
cl, ind = temp.remove_radius_outlier(nb_points=10, radius=0.1)
temp = temp.select_by_index(ind)
o3d.visualization.draw_geometries([temp], window_name="filterpoints", width=680, height=480)
points = np.asarray(temp.points)
colors = np.asarray(temp.colors)
im = xyz2depth(points, depth_cam_matrix, h,w, depth_scale=1000)
im.show()
im.save("F:\Gree\Data\living_room_eval/living_room2_png/cloud_bin_0.png")
print("shape",im.width, im.height)