# coding=utf-8
import numpy as np
from argparse import Namespace
import cv2
from PIL import Image
import open3d as o3d


# imd = Image.open("depthdemo0.png")
def xyz2depth(points, depth_cam_matrix, h,w, depth_scale=1000):
    #https://blog.csdn.net/qq_24815615/article/details/113276627
    for i in range(points.shape[0]):
        xyz = points[i]
        vu = xyz2vu(xyz, depth_cam_matrix)
        # xyz_col = colors[i] * 255
        depth = xyz[2] * depth_scale
        uv = np.array([[vu[1], vu[0]]])
        try:
            pix_rgb = im.getpixel((uv[0][0], uv[0][1]))
        except Exception as e:
            print("image index out of range", (uv[0][0], uv[0][1]))
            continue
        x = int(uv[0][0])
        y = int(uv[0][1])
        depth = int (depth)
        # r=g=b = depth
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

# fx, fy, cx, cy = 481.20, -480.0, 319.50, 239.50  # Robust indoorhttps://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
# fx, fy, cx, cy = 475.9, 475.9, 315.5, 245.4  # intel sr300
fx, fy, cx, cy = 367.28, 367.28, 255.16, 211.82 #kinectv2
# fx, fy, cx, cy = 589.3, 589.8, 321.14, 235.56  # kinectv1
# camera_matrix = {'xc': 319.50, 'zc': 239.50, 'fx': 481.2,'fy':480}#Robust indoor
camera_matrix = {'xc': 255.16, 'zc': 211.82, 'fx': 367.28,'fy':367.28}#kinectv2
camera_matrix = Namespace(**camera_matrix)
depth_cam_matrix = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
w,h = 512,424
im = Image.new('I', (w, h))
temp = o3d.io.read_point_cloud("kinect2022052102_1.ply")#camera121dpt_rgb425-512-012701.ply  camera222dpt_rgb425-512-012701.ply
o3d.visualization.draw_geometries([temp], window_name="oldpoints", width=800, height=600)
# filter
cl, ind = temp.remove_radius_outlier(nb_points=10, radius=0.1)
temp = temp.select_by_index(ind)
o3d.visualization.draw_geometries([temp], window_name="filterpoints", width=800, height=600)
points = np.asarray(temp.points)
colors = np.asarray(temp.colors)
im = xyz2depth(points, depth_cam_matrix, h,w, depth_scale=1000)
im.show()
im.save("kinect2022052102_1depth.png")
print("kinect20220521012",im.width, im.height)