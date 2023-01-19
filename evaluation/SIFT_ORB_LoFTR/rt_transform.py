import numpy as np
from utils.utils import C,A
import open3d as o3d
from copy import deepcopy
def rigid_transform_3D(A, B):
   """
    param:
        A: 源点云
        B: 目标点云
    return:
        rt: 源点云到目标点云的变换矩阵
   """
   assert len(A) == len(B)

   N = A.shape[0]  # total points
   centroid_A = np.mean(A, axis=0)
   centroid_B = np.mean(B, axis=0)

   # centre the points
   AA = A - np.tile(centroid_A, (N, 1))
   BB = B - np.tile(centroid_B, (N, 1))

   H = np.matmul(np.transpose(AA), BB)
   U, S, Vt = np.linalg.svd(H)
   R = np.matmul(Vt.T, U.T)

   # special reflection case
   if np.linalg.det(R) < 0:
       # print("Reflection detected")
       Vt[2, :] *= -1
       R = np.matmul(Vt.T, U.T)

   T = -np.matmul(R, centroid_A) + centroid_B
   RT = np.eye(4)
   RT[:3, :3] = R
   RT[:3, 3] = T
   RT[3, 3] = 1
   return RT
def rtRegistrat(superglueTxtpath,sourcePly,targetPly):
    # ------------------------------------------------------------------------
    # 配准测试数据集
    # ------------------------------------------------------------------------
    sourcePcd = o3d.io.read_point_cloud(sourcePly)
    targetPcd = o3d.io.read_point_cloud(targetPly)
    # ------------------------------------------------------------------------
    # 配准算法
    # A 固定两个点
    # C 固定三个点
    # livingroom数据集需要将lps、lpt乘以10
    # ------------------------------------------------------------------------
    index, lps, lpt = C(superglueTxtpath)
    rt = rigid_transform_3D(lps, lpt)
    print("rt", rt)
    source_pcd_rt = sourcePcd.transform(rt)
    finall_pcd = targetPcd+ source_pcd_rt
    o3d.visualization.draw_geometries([finall_pcd],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
if __name__=="__main__":
    superglueTxtpath = "F:/Gree/Data/scannet_eval/scene0002_00/superglueOutput/text3d/13_15.txt"
    sourcePly = "F:/Gree/Data/scannet_eval/scene0002_00/scene0002_00_evaluation/ply_eval/13.ply"
    targetPly = "F:/Gree/Data/scannet_eval/scene0002_00/scene0002_00_evaluation/ply_eval/15.ply"
    rtRegistrat(superglueTxtpath,sourcePly,targetPly)





