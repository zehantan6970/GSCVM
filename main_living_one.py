from main_living_all import *

if __name__=="__main__":
    name_s=str(439)                                                     #改，source图的名称
    name_t=str(499)                                                     #改，terget图的名称
    rt_all = []
    p_rt_all = []
    w_all = []
    path_o = '/home/wzh/supergan/tum-data/'                             #改tum路径
    data_path = '/home/wzh/supergan/tum-data/living_room_png/associations.txt' #，livingroom的associations.txt的路径
    fx, fy, cx, cy = 481.20, -480.0, 319.50, 239.50                     #livingname的内参
    path_match = '/home/wzh/supergan/tum-data/living_room_png/'         #改 #改，living_room_png的路径+‘/’
    ply_nlist, rgb_nlist = livingroom_ply_to_rgb('/home/wzh/supergan/tum-data/living_room_png/ply_to_png')#改路径
    pcd=o3d.io.read_point_cloud(path_o + "living_room_png/ply/" + str(ply_nlist[rgb_nlist.index(int(name_s))]) + ".ply")
    pcd_T = o3d.io.read_point_cloud(path_o + "living_room_png/ply/" + str(ply_nlist[rgb_nlist.index(int(name_t))  ]) + ".ply")                                    # 找到当前帧得到的对应文件
    lpsn, lptn = gan(name_s, name_s, name_t, name_t, fx, fy, cx, cy, path_match,super_point=40)   #改 superglue生成多少个点
    path = path_o + 'textliving/' + name_s + "_" + name_t + ".txt"                                # 漏斗所读取txt路径
    dict_vote, the_index, lps, lpt = A(path)                                                      #choose B(path), or C(path)
    rt, rt_all = choice_number_rtpoints(100, dict_vote, the_index, lptn, lpsn, rt_all)  # 改,筛点的阈值
    p_rt_all = TransformPointCloud(pcd_T, rt_all, p_rt_all)
    print('-' * 150)
    WriteRt(rt_all, './output/TumOneRt.txt', w_all)  # 改,是否写入rt转七元数
    WriteCameraXYZ(p_rt_all, './output/TumOneCameraxyz.txt')  # 改,是都记录相机坐标原点变化
    final = pcd
    for h in range(len(p_rt_all)):  # 把第一帧分别拼接之前算出的像第一帧转化的点云
        final += p_rt_all[h]
    o3d.io.write_point_cloud("super-living499-439.ply", final, write_ascii=True)  #改 储存点云并命名
    o3d.visualization.draw_geometries([pcd],
                                      window_name="pcd",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
    o3d.visualization.draw_geometries([pcd_T],
                                      window_name="pcd_T",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)

    o3d.visualization.draw_geometries([final],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)