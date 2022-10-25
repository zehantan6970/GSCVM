import json
import copy
import argparse
from easydict import EasyDict as edict
from models.PointDSC import PointDSC
from utils.pointcloud import estimate_normal
import torch
import numpy as np
import open3d as o3d 
import logging
import time
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关  此时是INFO

# 第二步，创建一个handler，用于写入日志文件
logfile = '/home/light/gree/align/PointDSC-master/fcgf.log'
fh = logging.FileHandler(logfile, mode='w')  # open的打开模式这里可以进行参考
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

# 第三步，再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)   # 输出到console的log等级的开关

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
def extract_fcgf_features(pcd_path, downsample, device, weight_path='misc/ResUNetBN2C-feat32-3dmatch-v0.05.pth'):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    raw_src_pcd = raw_src_pcd.voxel_down_sample(voxel_size=downsample)
    pts = np.array(raw_src_pcd.points)
    from misc.fcgf import ResUNetBN2C as FCGF
    from misc.cal_fcgf import extract_features
    fcgf_model = FCGF(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=7,
        normalize_feature=True
    ).to(device)
    checkpoint = torch.load(weight_path)
    fcgf_model.load_state_dict(checkpoint['state_dict'])
    fcgf_model.eval()

    xyz_down, features = extract_features(
        fcgf_model,
        xyz=pts,
        rgb=None,
        normal=None,
        voxel_size=downsample,
        skip_check=True,
    )
    return raw_src_pcd, xyz_down.astype(np.float32), features.detach().cpu().numpy()

def extract_fpfh_features(pcd_path, downsample, device):
    raw_src_pcd = o3d.io.read_point_cloud(pcd_path)
    # 为两个点云分别进行outlier removal
    # raw_src_pcd=raw_src_pcd.uniform_down_sample(every_k_points=40)
    raw_src_pcd = raw_src_pcd.voxel_down_sample(voxel_size=downsample)
    estimate_normal(raw_src_pcd, radius=downsample*2)
    src_pcd = raw_src_pcd.voxel_down_sample(downsample)
    src_features = o3d.pipelines.registration.compute_fpfh_feature(src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=downsample * 5, max_nn=100))
    src_features = np.array(src_features.data).T
    src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
    return raw_src_pcd, np.array(src_pcd.points), src_features

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not source_temp.has_normals():
        estimate_normal(source_temp)
        estimate_normal(target_temp)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # o3d.io.write_point_cloud("499_439PointDscFpfh.ply",(source_temp+target_temp))

def generate_rt(source_ply,target_ply,descriptor='fcgf'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='PointDSC_3DMatch_release', type=str, help='snapshot dir')
    parser.add_argument('--pcd1', default=source_ply, type=str)
    parser.add_argument('--pcd2', default=target_ply, type=str)
    parser.add_argument('--descriptor', default=descriptor, type=str, choices=['fcgf', 'fpfh'])
    parser.add_argument('--use_gpu', default=True, type=str2bool)
    args = parser.parse_args()

    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)
    spcd=o3d.io.read_point_cloud(source_ply)
    tpcd=o3d.io.read_point_cloud(target_ply)
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = PointDSC(
        in_dim=config.in_dim,
        num_layers=config.num_layers,
        num_channels=config.num_channels,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        sigma_d=config.sigma_d,
        k=config.k,
        nms_radius=config.inlier_threshold,
    ).to(device)
    miss = model.load_state_dict(
        torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl', map_location=device), strict=False)
    print(miss)
    model.eval()

    # extract features
    if args.descriptor == 'fpfh':
        raw_src_pcd, src_pts, src_features = extract_fpfh_features(args.pcd1, config.downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fpfh_features(args.pcd2, config.downsample, device)
    else:
        raw_src_pcd, src_pts, src_features = extract_fcgf_features(args.pcd1, config.downsample, device)
        raw_tgt_pcd, tgt_pts, tgt_features = extract_fcgf_features(args.pcd2, config.downsample, device)

    # matching
    distance = np.sqrt(2 - 2 * (src_features @ tgt_features.T) + 1e-6)
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)
    src_keypts = src_pts[corr[:, 0]]
    tgt_keypts = tgt_pts[corr[:, 1]]
    corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=-1)
    corr_pos = corr_pos - corr_pos.mean(0)

    # outlier rejection
    data = {
        'corr_pos': torch.from_numpy(corr_pos)[None].to(device).float(),
        'src_keypts': torch.from_numpy(src_keypts)[None].to(device).float(),
        'tgt_keypts': torch.from_numpy(tgt_keypts)[None].to(device).float(),
        'testing': True,
    }
    res = model(data)

    # First plot the original state of the point clouds
    # draw_registration_result(raw_src_pcd, raw_tgt_pcd, np.identity(4))

    # Plot point clouds after registration
    print(res['final_trans'][0].detach().cpu().numpy())
    # draw_registration_result(spcd, tpcd, res['final_trans'][0].detach().cpu().numpy())
    return res['final_trans'][0].detach().cpu().numpy()
if __name__ == '__main__':
    from config import str2bool
    gt_log = "pairs.txt"
    allTime = 0
    with open(gt_log, mode="r") as f:
        lines = f.readlines()
        num=len(lines)
        for i, line in enumerate(lines):
            id_lst = line.strip().split(" ")
            id_lst[0] = id_lst[0][:-4]
            id_lst[1] = id_lst[1][:-4]
            txt_name = str(id_lst[0]) + "_" + str(id_lst[1]) + ".txt"
            sourcePly = "./ply_eval/" + str(id_lst[1]
                ) + ".ply"
            targetPly = "./ply_eval/" + str(id_lst[0]) + ".ply"
            timeStart=time.time()
            rt = generate_rt(sourcePly, targetPly)
            timeEnd=time.time()
            allTime += (timeEnd - timeStart)
            meanTime = allTime / (i + 1)
            print("mean time:", meanTime)
            logging.info(str(id_lst[0]) + " " * 8 + str(id_lst[1]) + " " * 7 + str(num))
            for col in rt:
                logging.info(str(col[0]) + " " * 2 + str(col[1]) + " " * 2 + str(col[2]) + " " * 2 + str(col[3]))

