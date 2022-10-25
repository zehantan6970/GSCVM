import open3d
import tensorflow as tf
import numpy as np
import os
import copy
import time
from utils.config import Config
from datasets.common import Dataset
from models.KPFCNN_model import KernelPointFCNN
import logging
from demo_3dmatch_registration import MiniDataset,RegTester,execute_global_registration,draw_registration_result
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关  此时是INFO

# 第二步，创建一个handler，用于写入日志文件
logfile = 'subReconstruction10.log'
# 这里可以进行参考open的打开模式
fh = logging.FileHandler(logfile, mode='w')
# 输出到file的log等级的开关
fh.setLevel(logging.CRITICAL)

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
# logger.addHandler(ch)

# 日志级别
# logger.debug('debug级别，一般用来打印一些调试信息，级别最低')
# logger.info('info级别，一般用来打印一些正常的操作信息')
# logger.warning('waring级别，一般用来打印警告信息')
# logger.error('error级别，一般用来打印一些错误信息')
# logger.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')

# DEBUG：详细的信息,通常只出现在诊断问题上
# INFO：确认一切按预期运行
# WARNING（默认）：一个迹象表明,一些意想不到的事情发生了,或表明一些问题在不久的将来(例如。磁盘空间低”)。这个软件还能按预期工作。
# ERROR：更严重的问题,软件没能执行一些功能
# CRITICAL：一个严重的错误,这表明程序本身可能无法继续运行
def generateRt(sourcePly,targetPly):
    sourcenpz = sourcePly[:-4]+ ".npz"
    targetnpz = targetPly[:-4] + ".npz"
    point_cloud_files = [sourcePly, targetPly]  # source is ply_eval/4.ply target is ply_eval/0.ply
    path = 'results/Log_contraloss/'
    config = Config()
    config.load(path)

    # Initiate dataset configuration
    dataset = MiniDataset(files=point_cloud_files, voxel_size=0.2)

    # Initialize input pipelines
    dataset.init_test_input_pipeline(config)

    model = KernelPointFCNN(dataset.flat_inputs, config)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    chosen_step = np.sort(snap_steps)[-1]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))
    tester = RegTester(model, restore_snap=chosen_snap)

    # calculate descriptors
    tester.generate_descriptor(model, dataset)

    # Load the descriptors and estimate the transformation parameters using RANSAC
    src_pcd = open3d.read_point_cloud(sourcePly)
    src_data = np.load(sourcenpz)
    src_features = open3d.registration.Feature()
    src_features.data = src_data["features"].T
    src_keypts = open3d.PointCloud()
    src_keypts.points = open3d.Vector3dVector(src_data["keypts"])
    src_scores = src_data["scores"]

    tgt_pcd = open3d.read_point_cloud(targetPly)
    tgt_data = np.load(targetnpz)
    tgt_features = open3d.registration.Feature()
    tgt_features.data = tgt_data["features"].T
    tgt_keypts = open3d.PointCloud()
    tgt_keypts.points = open3d.Vector3dVector(tgt_data["keypts"])
    tgt_scores = tgt_data["scores"]
    result_ransac = execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features, 0.05)
    # First plot the original state of the point clouds
    # draw_registration_result(src_pcd, tgt_pcd, np.identity(4))#dispaly

    # Plot point clouds after registration
    # draw_registration_result(src_pcd, tgt_pcd, result_ransac.transformation)  # dispaly
    rt = result_ransac.transformation
    return rt

def subReconstruction(fragmentsPath,sub=10):
    """
        以sub为间隔进行点云重建
        param:
            fragmentsPath: 需要重建的所有ply保存的文件夹路径
            sub: 跳跃间隔
    """
    # 设置拼接点云片段数量
    allfragmentsNum = 881
    # 设置初始rt矩阵
    Rt = np.identity(4)
    fragmentsNum=0
    # 读取第0帧点云
    Pcd0 = open3d.io.read_point_cloud(fragmentsPath + str(0) + ".ply")
    allTime=0
    # 记录遍历次数
    i=0
    while fragmentsNum <=allfragmentsNum:
        print("start")
        print("即将拼接第%d张点云",fragmentsNum)
        i+=1
        if fragmentsNum - sub > 0:
            sourcePly = fragmentsPath + str(fragmentsNum) + ".ply"
            targetPly = fragmentsPath + str(fragmentsNum - sub) + ".ply"
        else:
            sourcePly = fragmentsPath + str(fragmentsNum) + ".ply"
            targetPly = fragmentsPath + str(0) + ".ply"
        timeStart = time.time()
        # 生成source到target的rt
        rt = generateRt(sourcePly, targetPly)
        timeEnd=time.time()
        allTime += (timeEnd - timeStart)
        meanTime = allTime / (i + 1)
        print("mean time:", meanTime)
        # fragmentsNum到0的Rt矩阵
        Rt = Rt @ rt
        # --------------------------------------------------------------
        # 将Rt保存为log文件
        # --------------------------------------------------------------
        idx1, idx2 = "0", str(fragmentsNum)
        logging.critical(idx1 + " " * 8 + str(fragmentsNum) + " " * 7 + "881")
        for col in Rt:
            logging.critical(str(col[0]) + " " * 2 + str(col[1]) + " " * 2 + str(col[2]) + " " * 2 + str(col[3]))
        PcdB = open3d.io.read_point_cloud(fragmentsPath+ str(fragmentsNum) + ".ply")
        Pcd0 += PcdB.transform(Rt)
        Pcd0 = open3d.voxel_down_sample(Pcd0,voxel_size=0.01)
        fragmentsNum += sub
        print("End")
        print("--------------------------------------------------------------------")
    # --------------------------------------------------------------
    # 显示重建结果并保存
    # --------------------------------------------------------------
    open3d.visualization.draw_geometries([Pcd0],
                                  window_name="final",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  )
    open3d.write_point_cloud("subReconstruction.ply",Pcd0)


if __name__ == "__main__":
    subReconstruction("/home/light/gree/datasets/livingRoom499_439/")
