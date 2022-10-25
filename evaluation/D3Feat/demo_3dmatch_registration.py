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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关  此时是INFO

# 第二步，创建一个handler，用于写入日志文件
logfile = '/home/light/gree/align/D3Feat/d3feat.log'
# open的打开模式这里可以进行参考
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
open3d.set_verbosity_level(open3d.VerbosityLevel.Error)


class MiniDataset(Dataset):
    def __init__(self, files, voxel_size=0.03):
        Dataset.__init__(self, 'Mini')
        self.num_test = 0
        self.anc_points = {"train": [], "test": []}
        self.ids_list = {"train": [], "test": []}
        for filename in files:
            pcd = open3d.read_point_cloud(filename)
            pcd = open3d.voxel_down_sample(pcd, voxel_size=voxel_size)
            points = np.array(pcd.points)/10
            self.anc_points['test'] += [points]
            self.ids_list['test'] += [filename]
            self.num_test += 1

    def get_batch_gen(self, split, config):
        def random_balanced_gen():
            # Initiate concatenation lists
            anc_points_list = []
            pos_points_list = []
            anc_keypts_list = []
            pos_keypts_list = []
            backup_anc_points_list = []
            backup_pos_points_list = []
            ti_list = []
            ti_list_pos = []
            batch_n = 0

            gen_indices = np.arange(self.num_test)

            for p_i in gen_indices:
                anc_id = self.ids_list['test'][p_i]
                pos_id = self.ids_list['test'][p_i]

                anc_ind = self.ids_list['test'].index(anc_id)
                pos_ind = self.ids_list['test'].index(pos_id)
                anc_points = self.anc_points['test'][anc_ind].astype(np.float32)
                pos_points = self.anc_points['test'][pos_ind].astype(np.float32)
                # back up point cloud
                backup_anc_points = anc_points
                backup_pos_points = pos_points

                n = anc_points.shape[0] + pos_points.shape[0]

                anc_keypts = np.array([])
                pos_keypts = np.array([])

                # Add data to current batch
                anc_points_list += [anc_points]
                anc_keypts_list += [anc_keypts]
                pos_points_list += [pos_points]
                pos_keypts_list += [pos_keypts]
                backup_anc_points_list += [backup_anc_points]
                backup_pos_points_list += [backup_pos_points]
                ti_list += [p_i]
                ti_list_pos += [p_i]

                yield (np.concatenate(anc_points_list + pos_points_list, axis=0),  # anc_points
                       np.concatenate(anc_keypts_list, axis=0),  # anc_keypts
                       np.concatenate(pos_keypts_list, axis=0),
                       np.array(ti_list + ti_list_pos, dtype=np.int32),  # anc_obj_index
                       np.array([tp.shape[0] for tp in anc_points_list] + [tp.shape[0] for tp in pos_points_list]),  # anc_stack_length
                       np.array([anc_id, pos_id]),
                       np.concatenate(backup_anc_points_list + backup_pos_points_list, axis=0)
                       )
                anc_points_list = []
                pos_points_list = []
                anc_keypts_list = []
                pos_keypts_list = []
                backup_anc_points_list = []
                backup_pos_points_list = []
                ti_list = []
                ti_list_pos = []

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.float32)
        gen_shapes = ([None, 3], [None], [None], [None], [None], [None], [None, 3])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):
        def tf_map(anc_points, anc_keypts, pos_keypts, obj_inds, stack_lengths, ply_id, backup_points):
            batch_inds = self.tf_get_batch_inds(stack_lengths)
            stacked_features = tf.ones((tf.shape(anc_points)[0], 1), dtype=tf.float32)
            anchor_input_list = self.tf_descriptor_input(config,
                                                         anc_points,
                                                         stacked_features,
                                                         stack_lengths,
                                                         batch_inds)
            return anchor_input_list + [stack_lengths, anc_keypts, pos_keypts, ply_id, backup_points]

        return tf_map


class RegTester:
    def __init__(self, model, restore_snap=None):

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)
            self.experiment_str = restore_snap.split("_")[-1][:8] + "-" + restore_snap.split("-")[-1]

        ## The release weight is pretrained on 3DMatch dataset, where we use a voxel downsample with 0.03m.
        ## If you want to test your own point cloud data which have different scale, you should change the scale
        ## of the kernel points so that the receptive field is also enlarged. For example, when testing the
        ## generalization on ETH, we use the following code to rescale the kernel points.

        # for v in my_vars:
        #     if 'kernel_points' in v.name:
        #         rescale_op = v.assign(tf.multiply(v, 0.10 / 0.03))
        #         self.sess.run(rescale_op)

    def generate_descriptor(self, model, dataset):
        self.sess.run(dataset.test_init_op)

        t = []
        for i in range(dataset.num_test):
            stime = time.time()
            ops = [model.anchor_inputs, model.out_features, model.out_scores, model.anc_id]
            [inputs, features, scores, anc_id] = self.sess.run(ops, {model.dropout_prob: 1.0})
            t += [time.time() - stime]

            # selecet keypoint based on scores
            scores_first_pcd = scores[inputs['in_batches'][0][:-1]]
            selected_keypoints_id = np.argsort(scores_first_pcd, axis=0)[:].squeeze()
            keypts_score = scores[selected_keypoints_id]
            keypts_loc = inputs['backup_points'][selected_keypoints_id]
            anc_features = features[selected_keypoints_id]
            np.savez_compressed(
                anc_id.decode("utf-8").replace('.ply', ''),
                keypts=keypts_loc,
                features=anc_features,
                scores=keypts_score,
            )


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    open3d.estimate_normals(source_temp)
    open3d.estimate_normals(target_temp)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])
    # open3d.write_point_cloud("499_439.ply",(source_temp+target_temp))


def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = open3d.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc,
        distance_threshold,
        open3d.TransformationEstimationPointToPoint(False), 4,
        [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        open3d.RANSACConvergenceCriteria(4000000, 500))
    return result


def plyRegirtration():
    # ----------------------------------------------------------------------------------
    # 生成自己算法估计的变换矩阵
    # 更换superglue生成的三维坐标的路径和gt.log的路径和自己的log文件路径，自己的log文件路径在第13行
    # ----------------------------------------------------------------------------------
    gt_log = "/home/light/gree/align/D3Feat/pairs.txt"
    allTime = 0
    with open(gt_log, mode="r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            id_lst = line.strip().split(" ")
            id_lst[0] = id_lst[0][:-4]
            id_lst[1] = id_lst[1][:-4]
            txt_name = str(id_lst[0]) + "_" + str(id_lst[1]) + ".txt"
            sourcePly = "/home/light/gree/align/D3Feat/ply_eval/" + str(
                id_lst[1]) + ".ply"
            targetPly = "/home/light/gree/align/D3Feat/ply_eval/" + str(
                id_lst[0]) + ".ply"
            # sourcePly="/home/gree/datasets/livingRoom499_439/499.ply"
            # targetPly="/home/gree/datasets/livingRoom499_439/439.ply"
            sourcenpz = "/home/light/gree/align/D3Feat/ply_eval/" + str(id_lst[1]) + ".npz"
            targetnpz = "/home/light/gree/align/D3Feat/ply_eval/" + str(id_lst[0]) + ".npz"
            # point_cloud_files = ["demo_data/cloud_bin_0.ply", "demo_data/cloud_bin_1.ply"]
            print(sourcePly, targetPly)
            print("Start")
            timeStart = time.time()
            point_cloud_files = [sourcePly, targetPly]  # source is ply_eval/4.ply target is ply_eval/0.ply
            # point_cloud_files = ["/home/tan/pycode/tf/D3Feat/ply_eval/7.ply", "/home/tan/pycode/tf/D3Feat/ply_eval/1.ply"]
            path = 'results/Log_contraloss/'
            config = Config()
            config.load(path)

            # Initiate dataset configuration
            dataset = MiniDataset(files=point_cloud_files, voxel_size=0.03)

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
            draw_registration_result(src_pcd, tgt_pcd, result_ransac.transformation)  # dispaly
            rt = result_ransac.transformation
            timeEnd = time.time()
            allTime += (timeEnd - timeStart)
            meanTime = allTime / (i + 1)
            print("mean time:", meanTime)
            print("End")
            idx1, idx2 = txt_name[:-4].split("_")[0], txt_name[:-4].split("_")[1]
            logging.critical(idx1 + " " * 8 + idx2 + " " * 7 + "52")
            for col in rt:
                logging.critical(str(col[0]) + " " * 2 + str(col[1]) + " " * 2 + str(col[2]) + " " * 2 + str(col[3]))
if __name__ == '__main__':
    plyRegirtration()

