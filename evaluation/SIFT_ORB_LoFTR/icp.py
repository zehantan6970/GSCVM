import open3d as o3d
import numpy as np

#读取电脑中的 ply 点云文件
# source = o3d.io.read_point_cloud("F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\0.ply")  #source 为需要配准的点云
# target = o3d.io.read_point_cloud("F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\4.ply")  #target 为目标点云
# o3d.io.write_point_cloud("F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\0.pcd", source)
# o3d.io.write_point_cloud("F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\4.pcd", target)
source = o3d.io.read_point_cloud("F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\0.pcd")  #source 为需要配准的点云
target = o3d.io.read_point_cloud("F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\4.pcd")
#为两个点云上上不同的颜色
# source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
# target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

#为两个点云分别进行outlier removal
processed_source, outlier_index1 = source.remove_radius_outlier(nb_points=16,
                                              radius=0.05)

processed_target, outlier_index2 = target.remove_radius_outlier(
                                              nb_points=16,
                                              radius=0.05)
print("run here")
threshold = 1.0  #移动范围的阀值
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                         [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                         [0,0,1,0],   # 这个矩阵为初始变换
                         [0,0,0,1]])

#运行icp
reg_p2p = o3d.pipelines.registration.registration_icp(
        processed_source, processed_target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

#将我们的矩阵依照输出的变换矩阵进行变换
print(reg_p2p.transformation)
processed_source.transform(reg_p2p.transformation)
finall_pcd=processed_source+processed_target
o3d.visualization.draw_geometries([finall_pcd],
                                  window_name="final",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
