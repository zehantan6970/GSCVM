import open3d as o3d
import numpy as np

'''
icp拼接livingroom数据集
'''
def every_nth(lst, nth):
    return lst[nth - 0::nth]


def getnames(path_d):  # 得到所取帧的文件名称（type:list）
    this_p_names = []
    with open(path_d, 'r') as f:
        for l in f.readlines():
            itemss = l.split()
            this_p_name = itemss[0]
            # print(this_p_name)
            this_p_names.append(this_p_name)
            names = every_nth(this_p_names, 5)  # 可改，隔几帧取ply
            names.insert(0, this_p_names[0])
    # print(names)
    return names
def livingroom_ply_to_rgb(livingroom_path):
    with open(livingroom_path, 'r') as f:

        lines = f.readlines()
        ply_n = []
        rgb_n = []
        a = len(lines)
        for i in range(0, len(lines), 2):
            # line = lines[i].strip("\n")
            # line = lines[i].split(" ")
            ply_n.append(int(lines[i].strip("\n").replace(' ', '')))

            rgb_n.append(int(lines[i + 1].strip("\n").replace(' ', '')))
        return ply_n,rgb_n
path_o='/home/wzh/supergan/tum-data/'                                   #改两个路径
data_path = '/home/wzh/supergan/tum-data/living_room_png/associations.txt'
names = getnames(data_path)  # 得到names,livingroom的点云文件和rgb不同名
print(names)
rt_all = []  # 储存所有rt
p_rt_all = []  # 储存所有被rt转换过的点云
if __name__=="__main__":
    name_s=str(439)                                                      #改
    name_t=str(499)
    path_o = '/home/wzh/supergan/tum-data/'
    data_path = '/home/wzh/supergan/tum-data/living_room_png/associations.txt'
    fx, fy, cx, cy = 481.20, -480.0, 319.50, 239.50                      #livingname的内参
    path_match = '/home/wzh/supergan/tum-data/living_room_png/'
    ply_nlist, rgb_nlist = livingroom_ply_to_rgb('/home/wzh/supergan/tum-data/living_room_png/ply_to_png')
    a_pcd = rgb_nlist.index(int(name_s))                                 #source的图片名称
    a_pct = rgb_nlist.index(int(name_t))                                 #terget的图片名称
    ply_pcd_name = ply_nlist[a_pcd]
    ply_pct_name = ply_nlist[a_pct]
    path_ply1 = path_o + "living_room_png/ply/" + str(ply_pcd_name) + ".ply"
    path_ply2 = path_o + "living_room_png/ply/" + str(ply_pct_name) + ".ply"
    pcd = o3d.io.read_point_cloud(path_ply1)
    pcd_T = o3d.io.read_point_cloud(path_ply2)  # 找到当前帧得到的对应文件
    threshold = 1.0  #移动范围的阀值
    trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                             [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                             [0,0,1,0],   # 这个矩阵为初始变换
                             [0,0,0,1]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_T, pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # rt_all.append(reg_p2p.transformation)
    import copy
    p_rt = copy.deepcopy(pcd_T)  # 对当前帧进行拷贝
    # for v in reversed(rt_all):  # 循环遍历当前rt_all里的所有rt
    p_rt = p_rt.transform(reg_p2p.transformation)  # 转          #对当前帧进行转换（倒序）
    # p_rt_all.append(p_rt)  # 将转换后的当前帧进行储存
    # for h in range(len(p_rt_all)):  # 把第一帧分别拼接之前算出的像第一帧转化的点云
    final = pcd+p_rt
    o3d.visualization.draw_geometries([final],
                                  window_name="final",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
    o3d.io.write_point_cloud("icp_living499-439.ply", final, write_ascii=True)  #改





















# # for z in range(len（names）-1):  # len（names）-1
# path = path_o + 'text/' + names[z] + "_" + names[z + 1] + ".txt"  # 漏斗所读取txt路径
# print(path)
# ply_nlist, rgb_nlist = livingroom_ply_to_rgb('/home/wzh/supergan/tum-data/living_room_png/ply_to_png')
# a_pcd = rgb_nlist.index(int(names[z]))
# a_pct = rgb_nlist.index(int(names[z + 1]))
# ply_pcd_name = ply_nlist[a_pcd]
# ply_pct_name = ply_nlist[a_pct]
# # path_ply1 = path_o + "data1/ply/" + names_ply[z] + '.ply'  # 历史帧点云文件路径
# # path_ply2 = path_o + "data1/ply/" + names_ply[z + 1] + '.ply'  # #下一帧点云文件路径
# path_ply1=path_o+"dataset/ply/"+names[z]+".ply"             #历史帧点云文件路径
# path_ply2=path_o+"dataset/ply/"+names[z+1]+".ply"#          #下一帧点云文件路径
# path_ply1 = path_o + "living_room_png/ply/" + str(ply_pcd_name) + ".ply"
# path_ply2 = path_o + "living_room_png/ply/" + str(ply_pct_name) + ".ply"
#
# pcd = o3d.io.read_point_cloud(path_ply1)  # 找到历史帧对应的点云文件
# pcd_T = o3d.io.read_point_cloud(path_ply2)  # 找到当前帧得到的对应文件
#
# threshold = 1.0  #移动范围的阀值
# trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
#                          [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
#                          [0,0,1,0],   # 这个矩阵为初始变换
#                          [0,0,0,1]])
# reg_p2p = o3d.pipelines.registration.registration_icp(
#         pcd_T, pcd, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint())
# rt_all.append(reg_p2p.transformation)
# import copy
# p_rt = copy.deepcopy(pcd_T)  # 对当前帧进行拷贝
# for v in reversed(rt_all):  # 循环遍历当前rt_all里的所有rt
#     p_rt = p_rt.transform(v)  # 转          #对当前帧进行转换（倒序）
# p_rt_all.append(p_rt)  # 将转换后的当前帧进行储存
# final = o3d.io.read_point_cloud(path_o + "living_room_png/ply/" + "167" + ".ply")
# # for h in range(len(p_rt_all)):  # 把第一帧分别拼接之前算出的像第一帧转化的点云
# final += p_rt
# o3d.visualization.draw_geometries([final],
#                               window_name="final",
#                               width=1024, height=768,
#                               left=50, top=50,
#                               mesh_show_back_face=False)
# o3d.io.write_point_cloud("icp_living.ply", final, write_ascii=True)