import numpy as np
import open3d as o3d
import argparse
import torch
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from config import make_cfg
from model import create_model

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npypath", required=True, help="path to point cloud numpy file")
    parser.add_argument("--plypath", required=True, help="path to point cloud ply file")
    parser.add_argument("--weights", required=True, help="model weights file")
    parser.add_argument("--frames", type=int, required=True, help="total number of points clouds")
    parser.add_argument("--skip", type=int, required=True, help="interval of matching point cloud id")
    return parser

def get_rt(args):
    rt_all = []
    for i in range(args.skip, args.frames, args.skip):
        sourcePly = args.npypath + str(i) + ".npy"  #"/home/zzw/GeoTransformer-main/data/newNpy/
        targetPly = args.npypath + str(i-args.skip) + ".npy"
        gt_file = "/home/zzw/GeoTransformer-main/data/heads/gt.npy"
        rt = get_result(sourcePly, targetPly, gt_file, args.weights)
        rt[0:3, 3] *= 10 #important!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        rt_all.append(rt)
        print('rt'+str(i)+'finish')
    return rt_all


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
    # print(estimated_transform)
    # transform = data_dict["transform"]
    return estimated_transform

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    rt_all = get_rt(args)
    pc_all = o3d.io.read_point_cloud(args.plypath + "0.ply")
    pc_all = pc_all.voxel_down_sample(voxel_size=0.2)
    for i in range(args.skip, args.frames, args.skip):
        pc = o3d.io.read_point_cloud(args.plypath + str(i) + ".ply")
        pc = pc.voxel_down_sample(voxel_size=0.2)
        for j in range(int(i/args.skip)-1, -1, -1):
            pc = pc.transform(rt_all[j])
        pc_all += pc
        print('finish'+str(i))
    o3d.visualization.draw_geometries([pc_all],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)




