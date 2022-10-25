import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os.path
# import skimage.io as io

def transformxyz(path,flag_save,plyname):
    temp = o3d.io.read_point_cloud(path)
    points = np.asarray(temp.points)
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    #按z向上 y向前 x向右调整 而Kinect 使用以 Kinect 为中心的笛卡尔坐标系统，Y 轴朝上，z轴朝前，X 轴朝左
    #https://kinect-tutorials-zh.readthedocs.io/zh_CN/latest/kinect1/3_PointCloud.html
    x = -1*x_points
    y = z_points
    z = y_points
    colors = np.asarray(temp.colors)
    aa = np.dstack((x, y, z))
    newpoints = aa.reshape(points.shape[0], points.shape[1])
    # 将 xyz值传给Open3D.o3d.geometry.PointCloud并保存点云
    if flag_save:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(newpoints)
        pcd.colors = o3d.utility.Vector3dVector(colors)  # 将颜色存入pcd
        o3d.io.write_point_cloud(plyname, pcd)
    return newpoints
#https://blog.csdn.net/tycoer/article/details/124508790?spm=1001.2014.3001.5501
#已知像素坐标 uv,  像素坐标对应的深度 depth, 相机内参 K, 求解 点云坐标

def uv2xyz(uv, K, depth):
    '''
    Args:
        uv: pixel coordinates shape (n, 2)
        K: camera instrincs, shape (3, 3)
        depth: depth values of uv, shape (n, 1)
    Returns: point cloud coordinates xyz, shape (n, 3)
    '''
    assert depth.ndim == 2, f'depth shape should be (n, 1) instead of {depth.shape}'
    assert uv.ndim == 2, f'uv shape should be (n, 2) instead of {uv.shape}'

    # Another form
    u = uv[:, 0]
    v = uv[:, 1]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = np.hstack((x.reshape(-1, 1) * depth, y.reshape(-1, 1) * depth, depth))
    return xyz

#https://blog.csdn.net/tycoer/article/details/107959873
#python - 点云坐标 投射至 像素坐标
def xyz2vu(xyz,depth_cam_matrix,depth_dist_coeff=np.zeros(5)):
        # 畸变类型带添加
        fx,fy=depth_cam_matrix[0,0],depth_cam_matrix[1,1]
        cx,cy=depth_cam_matrix[0,2],depth_cam_matrix[1,2]
        xyz=np.array(xyz).flatten()
        x,y=xyz[0]/xyz[2],xyz[1]/xyz[2]
        r2=x*x+y*y
        f=1+depth_dist_coeff[0]*r2+depth_dist_coeff[1]*r2*r2+depth_dist_coeff[1]*r2*r2*r2
        x*=f
        y*=f
        dx=x+2*depth_dist_coeff[2]*x*y+depth_dist_coeff[3]*(r2+2*x*x)
        dy=y+2*depth_dist_coeff[3]*x*y+depth_dist_coeff[2]*(r2+2*y*y)
        x,y=dx,dy
        u,v=x*fx+cx,y*fy+cy
        vu=np.int0([v,u])
        return vu

########获取图片指定像素点的像素 csdn下载
def getPngPix(pngPath = "color0.png",pixelX = 1,pixelY = 1):
    img_src = Image.open(pngPath)
    img_src = img_src.convert('RGBA')
    str_strlist = img_src.load()
    data = str_strlist[pixelX,pixelY]
    img_src.close()
    return data
#https://www.cnblogs.com/BambooEatPanda/p/9921446.html
def convertPNG(pngfile,outdir):
    # READ THE DEPTH
    im_depth = cv2.imread(pngfile)
    #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=15),cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    # im.show()
    #save image
    # im.save(os.path.join(outdir,os.path.basename(pngfile)))
    return im
# https://blog.csdn.net/qq_44813407/article/details/115315618
def got_RGB(img_path):
    img = Image.open(img_path)
    width, height = img.size
    img = img.convert('RGB')
    array = []
    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((i, j))
            # if r != 0:
                # print(r, b, g)
            rgb = (r, g, b)
            array.append(rgb)
    return array

if __name__ == '__main__':
    ########### 造数据 #######
    #https://stackoverflow.com/questions/47967503/kinect-v2-get-real-xyz-points-from-raw-depth-image-in-matlab-without-visionkinec
    uv = np.array([[100, 200]])  # 像素坐标点 (100, 200)
    depth = np.array([[0.5]])  # 像素坐标点 (100, 200) 对应的深度 0.5~4.5(一般单位为m)
    # fx, fy, cx, cy = 367.28, 367.28, 255.16, 211.82
    fx, fy, cx, cy = 481.20, -480.0, 319.50, 239.50
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    ###########################
    xyz = uv2xyz(uv, K, depth)
    print("xyz=",xyz)
    ###########python - 点云坐标 投射至 像素坐标##
    # fx, fy, cx, cy = 367.28, 367.28, 255.16, 211.82 #kinectv2
    # fx, fy, cx, cy = 367.28, 367.28, 255.16, 211.82  # kinectv1
    depth_cam_matrix = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    xyz = [-0.2037037, - 0.03703704,  0.5]
    vu = xyz2vu(xyz,depth_cam_matrix)
    print("vu=", vu)
    flag_save = True
    temp = o3d.io.read_point_cloud("F:\Gree\Data\living_room_eval\living_room_47\livingroom2-fragments-ply/cloud_bin_6.ply")#camera121dpt_rgb425-512-012701.ply  camera222dpt_rgb425-512-012701.ply kinect2022052101.ply
    o3d.visualization.draw_geometries([temp], window_name="oldpoints", width=800, height=600)
    # filter
    cl, ind = temp.remove_radius_outlier(nb_points=10, radius=0.1)
    temp = temp.select_by_index(ind)
    o3d.visualization.draw_geometries([temp], window_name="newpoints", width=800, height=600)
    points = np.asarray(temp.points)
    colors = np.asarray(temp.colors)
    w=640
    h=480
    im = Image.new('RGB', (w, h))
    for i in range(points.shape[0]):
        xyz = points[i]
        vu = xyz2vu(xyz, depth_cam_matrix)
        xyz_col = colors[i] * 255
        uv = np.array([[vu[1], vu[0]]])
        try:
            pix_rgb = im.getpixel((uv[0][0], uv[0][1]))
        except Exception as e:
            print("image index out of range", (uv[0][0], uv[0][1]))
            continue
        x = int(uv[0][0])
        y = int(uv[0][1])
        r = int(xyz_col[0])
        g = int(xyz_col[1])
        b = int(xyz_col[2])
        im.putpixel((x, y), (r, g, b))  # im.putpixel((10,20),(255,32,43))
    im.show()
    im.save("kinect2022052101_1rgb.png")
    print("shape",im.width, im.height)

