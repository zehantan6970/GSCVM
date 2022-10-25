from demo import get_result
import os
import open3d as o3d
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import logging
#from frozenDir import relativePath
import time
import torch
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda


from config import make_cfg
from model import create_model
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关  此时是INFO

# 第二步，创建一个handler，用于写入日志文件
logfile = "geo.log"
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
def icp(source,target):
    # 为两个点云上上不同的颜色
    # source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
    # target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

    # 为两个点云分别进行outlier removal
    # processed_source, outlier_index1 = source.remove_radius_outlier(nb_points=16,
    #                                                                 radius=0.05)
    #
    # processed_target, outlier_index2 = target.remove_radius_outlier(
    #     nb_points=16,
    #     radius=0.05)
    print("原始点云中点的个数为：", np.asarray(source.points).shape[0])
    # o3d.visualization.draw_geometries([pcd])
    print("使用边长为0.005的体素对点云进行下采样")
    processed_source = source.voxel_down_sample(voxel_size=0.001)
    print("下采样之后点的个数为：", np.asarray(processed_source.points).shape[0])
    processed_target = target.voxel_down_sample(voxel_size=0.001)
    print("run here")
    threshold = 1.0  # 移动范围的阀值
    trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                             [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                             [0, 0, 1, 0],  # 这个矩阵为初始变换
                             [0, 0, 0, 1]])

    # 运行icp
    reg_p2p = o3d.pipelines.registration.registration_icp(
        processed_source, processed_target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # 将我们的矩阵依照输出的变换矩阵进行变换
    print(reg_p2p.transformation)
    # processed_source.transform(reg_p2p.transformation)
    # finall_pcd = processed_source + processed_target
    # o3d.visualization.draw_geometries([finall_pcd],
    #                                   window_name="final",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)
    return reg_p2p.transformation
def selfDefine(superglue_eval_txts,gt_log):
    allTime=0
    with open(gt_log, mode="r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
                id_lst = line.strip().split(" ")
                id_lst[0]=id_lst[0][:-4]
                id_lst[1] = id_lst[1][:-4]
                txt_name=str(id_lst[0])+"_"+str(id_lst[1])+".txt"
                sourcePly = relativePath()+"/registration_evaluate/scannet_eval/scene0002_00/scene0002_00_evaluation/ply_eval/"+str(id_lst[1])+".ply"
                targetPly = relativePath()+"/registration_evaluate/scannet_eval/scene0002_00/scene0002_00_evaluation/ply_eval/"+str(id_lst[0])+".ply"
                sourcePcd = o3d.io.read_point_cloud(sourcePly)
                targetPcd = o3d.io.read_point_cloud(targetPly)
                # ----------------------------------------------------------------------------------
                # superglue算法
                # ----------------------------------------------------------------------------------
                print(txt_name)
                txt_path=superglue_eval_txts+txt_name
                timeStart=time.time()
                n,lps,lpt=B(txt_path)
                # livingRoom需要乘以10
                rt = rigid_transform_3D(lpt, lps)
                timeEnd=time.time()
                allTime+=(timeEnd-timeStart)
                # 计算生成rt平均消耗时间
                meanTime=allTime/(i+1)
                print("mean time:",meanTime)
                source_pcd_rt = sourcePcd.transform(rt)
                finall_pcd = targetPcd + source_pcd_rt
                o3d.visualization.draw_geometries([finall_pcd],
                                                  window_name="final",
                                                  width=1024, height=768,
                                                  left=50, top=50,
                                                  mesh_show_back_face=False)
                # ----------------------------------------------------------------------------------
                # icp算法
                # ----------------------------------------------------------------------------------
                # timeStart = time.time()
                # rt=icp(sourcePcd,targetPcd)
                # timeEnd = time.time()
                # allTime+=(timeEnd-timeStart)
                # # 计算生成rt平均消耗时间
                # meanTime=allTime/(i+1)
                # print("mean time:",meanTime)
                idx1,idx2=txt_name[:-4].split("_")[0],txt_name[:-4].split("_")[1]
                logging.info(idx1+" "*8+idx2+" "*7+"52")
                for col in rt:
                    logging.info(str(col[0])+" "*2+str(col[1])+" "*2+str(col[2])+" "*2+str(col[3]))




def load_data(src_file, ref_file, gt_file):
    src_points = np.load(src_file)
    ref_points = np.load(ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    transform = np.load(gt_file)

    data_dict["transform"] = transform.astype(np.float32)

    return data_dict





def get_result(src_file, ref_file, gt_file, weights):


    cfg = make_cfg()
    # prepare data
    data_dict = load_data(src_file, ref_file, gt_file)
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]


    return estimated_transform
def them():
    allTime = 0
    num=0
    for i in range(52):
        for j in range(i + 1, 52):
            num+=1
            id_lst = [i, j]
            txt_name = str(id_lst[0]) + "_" + str(id_lst[1]) + ".txt"
            sourcePly = "/home/zzw/GeoTransformer-main/data/heads/" + str(
                id_lst[1]) + ".npy"
            targetPly = "/home/zzw/GeoTransformer-main/data/heads/"  + str(
                id_lst[0]) + ".npy"

            # ----------------------------------------------------------------------------------
            gt_file = ""
            timeStart = time.time()
            rt=get_result(sourcePly,targetPly,gt_file)
            timeEnd = time.time()
            allTime+=(timeEnd-timeStart)
            # # 计算生成rt平均消耗时间
            meanTime = allTime / num
            print("mean time:", meanTime)
            idx1, idx2 = txt_name[:-4].split("_")[0], txt_name[:-4].split("_")[1]
            logging.info(idx1 + " " * 8 + idx2 + " " * 7 + "52")
            for col in rt:
                logging.info(str(col[0]) + " " * 2 + str(col[1]) + " " * 2 + str(col[2]) + " " * 2 + str(col[3]))
if __name__=="__main__":
    # ----------------------------------------------------------------------------------
    # 生成自己算法估计的变换矩阵
    # 更换superglue生成的三维坐标的路径和gt.log的路径和自己的log文件路径，自己的log文件路径在第13行
    # ----------------------------------------------------------------------------------
    them()
