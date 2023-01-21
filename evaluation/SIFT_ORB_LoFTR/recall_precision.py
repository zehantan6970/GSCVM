from utils.utils import A,B,C
import os
# import open3d as o3d
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import logging
from frozenDir import relativePath
import time
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关  此时是INFO

# 第二步，创建一个handler，用于写入日志文件
logfile = relativePath()+'/registration_evaluate/result/orb_B.log'
# open的打开模式这里可以进行参考
fh = logging.FileHandler(logfile, mode='w')
# 输出到file的log等级的开关
fh.setLevel(logging.DEBUG)

# 第三步，再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
# 输出到console的log等级的开关
ch.setLevel(logging.WARNING)

# 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
formatter = logging.Formatter("")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
#
# 第五步，将logger添加到handler里面
logger.addHandler(fh)
logger.addHandler(ch)

# 日志级别
# logger.debug('debug级别，一般用来打印一些调试信息，级别最低')
# logger.info('info级别，一般用来打印一些正常的操作信息')
# logger.warning('waring级别，一般用来打印警告信息')
# logger.error('error级别，一般用来打印一些错误信息')
# logger.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')

#
# DEBUG：详细的信息,通常只出现在诊断问题上
# INFO：确认一切按预期运行
# WARNING（默认）：一个迹象表明,一些意想不到的事情发生了,或表明一些问题在不久的将来(例如。磁盘空间低”)。这个软件还能按预期工作。
# ERROR：更严重的问题,软件没能执行一些功能
# CRITICAL：一个严重的错误,这表明程序本身可能无法继续运行
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
# def icp(source,target):
#     # 为两个点云上上不同的颜色
#     # source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
#     # target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色
#
#     # 为两个点云分别进行outlier removal
#     processed_source = source.voxel_down_sample(voxel_size=0.001)
#
#     processed_target = target.voxel_down_sample(voxel_size=0.001)
#     print("run here")
#     threshold = 1.0  # 移动范围的阀值
#     trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
#                              [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
#                              [0, 0, 1, 0],  # 这个矩阵为初始变换
#                              [0, 0, 0, 1]])
#
#     # 运行icp
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         processed_source, processed_target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint())
#
#     # 将我们的矩阵依照输出的变换矩阵进行变换
#     print(reg_p2p.transformation)
#     # processed_source.transform(reg_p2p.transformation)
#     # finall_pcd = processed_source + processed_target
#     # o3d.visualization.draw_geometries([finall_pcd],
#     #                                   window_name="final",
#     #                                   width=1024, height=768,
#     #                                   left=50, top=50,
#     #                                   mesh_show_back_face=False)
#     return reg_p2p.transformation
def them(superglue_eval_txts):
    """
        生成与gt.log相同的pred.log
        param:
            superglue_eval_txts: superGlue生成的text3d文件夹路径
    """
    allTime = 0
    num=0
    fragmentsNum=52
    for i in range(fragmentsNum):
        for j in range(i + 1, fragmentsNum):
            num+=1
            txt_name = str(i) + "_" + str(j) + ".txt"

            # ----------------------------------------------------------------------------------
            # superglue算法
            # ----------------------------------------------------------------------------------
            print(txt_name)
            txt_path = superglue_eval_txts + txt_name
            timeStart = time.time()
            try:
                n, lps, lpt = B(txt_path)  # livingRoom需要除以10
                rt = rigid_transform_3D(lpt, lps)
            except:
                rt=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            timeEnd = time.time()
            allTime += (timeEnd - timeStart)
            # 计算生成rt平均消耗时间
            meanTime = allTime / num
            print("mean time:", meanTime)
            # ----------------------------------------------------------------------------------
            # icp算法
            # ----------------------------------------------------------------------------------
            # timeStart = time.time()
            # rt=icp(sourcePcd,targetPcd)
            # timeEnd = time.time()
            # allTime+=(timeEnd-timeStart)
            # # 计算生成rt平均消耗时间
            # meanTime=allTime/(num+1)
            # print("mean time:",meanTime)
            # ----------------------------------------------------------------------------------
            # 使用rt矩阵进行拼接并且可视化
            # ----------------------------------------------------------------------------------
            # sourcePly = "/home/light/gree/Hypothesis/heads_evaluation/ply_eval/" + str(
            #     j) + ".ply"
            # targetPly = "/home/light/gree/Hypothesis/heads_evaluation/ply_eval/" + str(
            #     i) + ".ply"
            # sourcePcd = o3d.io.read_point_cloud(sourcePly)
            # targetPcd = o3d.io.read_point_cloud(targetPly)
            # source_pcd_rt = sourcePcd.transform(rt)
            # finall_pcd = targetPcd + source_pcd_rt
            # o3d.visualization.draw_geometries([finall_pcd],
            #                                   window_name="final",
            #                                   width=1024, height=768,
            #                                   left=50, top=50,
            #                                   mesh_show_back_face=False)
            # ----------------------------------------------------------------------------------
            # 保存log日志
            # ----------------------------------------------------------------------------------
            idx1, idx2 = txt_name[:-4].split("_")[0], txt_name[:-4].split("_")[1]
            logging.info(idx1 + " " * 8 + idx2 + " " * 7 + str(fragmentsNum))
            for col in rt:
                logging.info(str(col[0]) + " " * 2 + str(col[1]) + " " * 2 + str(col[2]) + " " * 2 + str(col[3]))
if __name__=="__main__":
    # ----------------------------------------------------------------------------------
    # 生成自己算法估计的变换矩阵
    # 更换superglue生成的三维坐标的路径和gt.log的路径和自己的log文件路径，自己的log文件路径在第13行
    # ----------------------------------------------------------------------------------
    superglue_eval_txts= "/home/light/gree/Hypothesis/head_output/text3dorb/"
    them(superglue_eval_txts)
