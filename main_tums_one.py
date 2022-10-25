from main_tums import *

if __name__=="__main__":
    if not os.path.exists('./output'):
        os.makedirs('./output')
    rt_all = []
    p_rt_all=[]
    w_all = []
    name_s = '4.700200'                                     #改，要检测的图片的名称
    name_t = '8.100222'                                     #改
    path_o = '/home/wzh/supergan/tum-data/'#改三个路径
    data_path = '/home/wzh/supergan/tum-data/datatest/associate.txt'  # names的路径
    path_match = '/home/wzh/supergan/tum-data/datatest/'              # 改数据集路径
    names, names_depth = getnames(data_path,jump=1)                   # 得到names、depthnames,jump是多少帧检测一次，等于1时不跳帧
    # fx, fy, cx, cy = 517.306408,-516.469215,318.643040,255.313989   # 改内参
    fx, fy, cx, cy = 622.28, -622.89, 642.33, 366.55                  # 改内参，azure Kinect dk
    DeleteWrongDepthPointsOne(path_match,name_s,name_t,names,names_depth)
    lpsn, lptn = gan(name_s,names_depth[names.index(name_s)],name_t,names_depth[names.index(name_t)],fx, fy, cx, cy,path_match,super_point=50)    #改def getpair(img1,img1dep, img2,img2dep,fx, fy, cx, cy,path):
    path = path_o + 'texttest/' + name_s + "_" + name_t + ".txt"  # 漏斗所读取txt路径
    pcd=product_pcd( path_match+'rgb/'+name_s+'.png',path_match+'depth/'+names_depth[names.index(name_s)]+'.png',fx, fy, cx, cy)    #rgbpicturepath,depthpicturepath,fx, fy, cx, cy):
    pcd_T=product_pcd( path_match+'rgb/'+name_t+'.png',path_match+'depth/'+names_depth[names.index(name_t)]+'.png',fx, fy, cx, cy)
    dict_vote, the_index, lpsn, lptn = A(path)                        # 索引、历史帧点坐标type：二维数组
    print(dict_vote)
    rt, rt_all = choice_number_rtpoints(100,dict_vote,the_index,lptn, lpsn,rt_all)  # 改,筛点的阈值
    p_rt_all = TransformPointCloud(pcd_T,rt_all,p_rt_all)
    print('-' * 150)
    WriteRt(rt_all, './output/TumOneRt.txt', w_all)  # 改,是否写入rt转七元数
    WriteCameraXYZ(p_rt_all, './output/TumOneCameraxyz.txt')  # 改,是都记录相机坐标原点变化
    # final = product_pcd(path_match + 'rgb/' + names[0] + '.png', path_match + 'depth/' + names_depth[0] + '.png', fx,
    #                     fy, cx, cy)  # 生成点云rgbpicturepath,depthpicturepath,fx, fy, cx, cy):

    final =pcd
    for h in range(len(p_rt_all)):  # 把第一帧分别拼接之前算出的像第一帧转化的点云
        final += p_rt_all[h]

o3d.io.write_point_cloud("tumtest.ply", pcd, write_ascii=True)    #改
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
