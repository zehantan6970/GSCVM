import open3d as o3d
import numpy as np
from copy import deepcopy
from utils.rotateQuaternion import quat_to_pos_matrix_hm
def livingroom2_882():
    with open("C:/Users/asus/Desktop/Hypothesis/utils/livingRoom2.gt.freiburg",mode="r") as r:
        lines=r.readlines()
        ply = "F:/Gree/Data/living_room_eval/living_room_traj2_frei_png/ply/" + str(0) + ".ply"
        pcd = o3d.io.read_point_cloud(ply)
        siyuanshu = list(map(float, lines[0].strip().split()))
        tx, ty, tz, qx, qy, qz, qw = siyuanshu[1:]
        rt = quat_to_pos_matrix_hm(tx, ty, tz, qx, qy, qz, qw)
        pcdRt0 = pcd.transform(rt)
        for i in range(200,300):
            print(i)
            ply="F:/Gree/Data/living_room_eval/living_room_traj2_frei_png/ply/"+str(i)+".ply"
            pcd=o3d.io.read_point_cloud(ply)
            siyuanshu=list(map(float,lines[i].strip().split()))
            tx, ty, tz, qx, qy, qz, qw = siyuanshu[1:]
            rt= quat_to_pos_matrix_hm(tx, ty, tz, qx, qy, qz, qw)
            pcdRt = pcd.transform(rt)
            pcdRt0+=pcdRt
        r.close()
    o3d.visualization.draw_geometries([pcdRt0],
                                     window_name="final",
                                     width=1024, height=768,
                                     left=50, top=50,
                                     mesh_show_back_face=False)
 #    rt0=np.array([[ 1. ,   0. ,   0. ,   0.  ],
 # [ 0.  ,  1.  ,  0.   , 0.  ],
 # [ 0. ,   0. ,   1.  , -2.25],
 # [ 0.  ,  0. ,   0. ,   1.  ]]
 #                    )
 #    rt230=np.array([[ 0.11082713, -0.0263528,   0.99349076 ,-0.76478   ],
 # [ 0.06125402,  0.99792902,  0.01963742 ,-0.127117  ],
 # [-0.99195076 , 0.0586789  , 0.11221182 ,-0.782991  ],
 # [ 0.       ,   0.     ,     0.    ,      1.  ,      ]])
 #    source_pcd_rt = pcdA.transform(rt0)
 #    target_pcd_rt=pcdB.transform(rt230)
 #    finall_pcd = target_pcd_rt + source_pcd_rt
 #    o3d.visualization.draw_geometries([finall_pcd],
 #                                      window_name="final",
 #                                      width=1024, height=768,
 #                                      left=50, top=50,
 #                                      mesh_show_back_face=False)
# 测试准确
def rgbd_frames():
    ply_57 = "C:/Users/asus/Desktop/Hypothesis/evaluate/3dmatch/3dmatch-toolbox-master/3dmatch-toolbox-master/data/sample/depth-fusion-demo/rgbd-frames/ply/frame-000057.ply"
    ply_8="C:/Users/asus/Desktop/Hypothesis/evaluate/3dmatch/3dmatch-toolbox-master/3dmatch-toolbox-master/data/sample/depth-fusion-demo/rgbd-frames/ply/frame-000008.ply"
    ply_27 = "C:/Users/asus/Desktop/Hypothesis/evaluate/3dmatch/3dmatch-toolbox-master/3dmatch-toolbox-master/data/sample/depth-fusion-demo/rgbd-frames/ply/frame-000027.ply"
    pcd_57 = o3d.io.read_point_cloud(ply_57)
    pcd_8=o3d.io.read_point_cloud(ply_8)
    pcd_27 = o3d.io.read_point_cloud(ply_27)
    pose_57=np.array( [[8.6824435e-001	,  3.4352240e-001,	 -3.5782585e-001	, -5.7293946e-001]	,
                       [-3.3708364e-001	,  9.3781054e-001	,  8.2405970e-002	,  2.4003703e-002],
                       [3.6389530e-001	,  4.9070477e-002	,  9.3007708e-001,	  3.9424694e-001],
                       [0.0000000e+000	 , 0.0000000e+000	 , 0.0000000e+000	,  1.0000000e+000]]
                    )
    pose_8=np.array(  [[9.0709007e-001	,  2.7616704e-001	, -3.1753206e-001,	 -3.4287909e-001],
                       [-2.7506533e-001,	  9.6011728e-001,	  4.9265359e-002,	  1.2549059e-002],
                       [3.1848383e-001,	  4.2655181e-002	,  9.4690639e-001,	  2.9991144e-001],
                       [0.0000000e+000	,  0.0000000e+000	,  0.0000000e+000	,  1.0000000e+000]]
                        )
    pose_27 = np.array(  [[8.9126843e-001,	  2.9982448e-001,	 -3.4007171e-001,	 -3.7568077e-001],
                          [-2.9855296e-001	,  9.5261735e-001	,  5.7419233e-002	,  6.0497252e-003]	,
                          [3.4118575e-001	,  5.0355326e-002	,  9.3858135e-001	,  3.1213170e-001],
                          [0.0000000e+000	,  0.0000000e+000	,  0.0000000e+000	,  1.0000000e+000]]
                       )
    pcd_57_rt = pcd_57.transform(pose_57)
    pcd_8_rt=pcd_8.transform(pose_8)
    pcd_27_rt = pcd_27.transform(pose_27)
    finall_pcd = pcd_8_rt+pcd_27_rt+pcd_57_rt
    o3d.visualization.draw_geometries([finall_pcd],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
def scannet0002():
    ply_0="F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\0.ply"
    ply_25 = "F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\25.ply"
    ply_51 = "F:\\Gree\\Data\\scannet_eval\\scene0002_00\\scene0002_00_evaluation\\ply_eval\\51.ply"
    pcd_0 = o3d.io.read_point_cloud(ply_0)
    pcd_25 = o3d.io.read_point_cloud(ply_25)
    pcd_51=o3d.io.read_point_cloud(ply_51)
    pose_0=np.array([[-0.877021 ,0.121711 ,-0.464779 ,3.92242] ,
                    [0.46491, 0.459041, -0.75706 ,3.2302 ],
                    [0.12121 ,-0.880038, -0.459173,1.74705 ],
                    [0, 0 ,0 ,1]]
                    )
    pose_51=np.array([[0.216637 ,0.101106 ,-0.971003, 3.52993 ],
                    [0.954059 ,-0.232801 ,0.188616, 1.07624 ],
                    [-0.20698 ,-0.967254 ,-0.146894 ,1.90159 ],
                    [0 ,0, 0, 1 ]]
                    )
    pose_25=np.array([[-0.385036 ,0.68149, -0.622349 ,2.74395],
                    [0.922447 ,0.305333 ,-0.236352 ,4.45787 ],
                    [0.0289523 ,-0.665088 ,-0.746203,1.65205 ],
                    [0 ,0, 0 ,1 ]]
                    )
    pcd_51_rt = pcd_51.transform(pose_51)
    pcd_25_rt = pcd_25.transform(pose_25)
    pcd_0_rt = pcd_0.transform(pose_0)
    finall_pcd = pcd_0_rt+pcd_51_rt+pcd_25_rt
    o3d.visualization.draw_geometries([finall_pcd],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
def livingroom2_2350():
    plyA = "F:\\Gree\\Data\\living_room_eval\\living_room_2350\\livingroom2-ply\\00000.ply"
    plyB = "F:\\Gree\\Data\\living_room_eval\\living_room_2350\\livingroom2-ply\\00049.ply"
    pcdA = o3d.io.read_point_cloud(plyA)
    pcdB = o3d.io.read_point_cloud(plyB)
    pose_0 = np.array([[0.9918944425900298, 0.04345865564545356 ,-0.11940167506573783, -1.7038232039817223],
                        [7.38421665810668e-18, -0.9396926207859083 ,-0.3420201433256686 ,1.2380193956460008],
                        [-0.12706460860134952 ,0.33924787941857615 ,-0.9320758883004028 ,0.5648534678511549],
                        [0.0, 0.0, 0.0, 1.0]]
                      )
    pose_10=np.array([[0.9984847764308222, 0.017197949385689706 ,-0.05227218928664069 ,-1.5616840789611504],
                        [3.6524000700317015e-18 ,-0.9499087957041633, -0.3125272465623859, 1.3454274387108112],
                        [-0.055028640142121804 ,0.31205369791238435 ,-0.9484694715083429, 0.5504719937482865],
                        [0.0 ,0.0,0.0, 1.0]])
    pose_49 = np.array([[0.982551790395577 ,-0.07218533259105273 ,0.1714096174348601, -1.8715322492951203],
                        [0.0 ,-0.9216106401651888 ,-0.38811574038463165 ,1.9218643885604831],
                        [0.1859891910580988, 0.3813438155956248, -0.9055301845419205 ,0.5122516896462468],
                        [0.0 ,0.0 ,0.0 ,1.0]]
                       )
    pcdBrt = pcdB.transform(pose_49)
    pcdArt = pcdA.transform(pose_0)
    finall_pcd =  pcdArt+pcdBrt
    o3d.visualization.draw_geometries([finall_pcd],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
def head():
    plyA = "C:/Users/asus/Desktop/Hypothesis/registration_evaluate/heads_eval/heads/heads_evaluation/ply_eval/0.ply"
    plyB = "C:/Users/asus/Desktop/Hypothesis/registration_evaluate/heads_eval/heads/heads_evaluation/ply_eval/40.ply"
    pcdA = o3d.io.read_point_cloud(plyA)
    pcdB = o3d.io.read_point_cloud(plyB)
    pose_0 = np.array([  [9.3124521e-001,	  1.7166864e-002	, -3.6325571e-001	, -1.4959294e-001]	,
  [7.4632928e-002	,  9.6838450e-001	,  2.3717017e-001	, -1.3273862e-001]	,
  [3.5591775e-001	, -2.4801806e-001	,  9.0070820e-001	,  1.8034773e-001]	,
  [0.0000000e+000	,  0.0000000e+000	,  0.0000000e+000	,  1.0000000e+000]]
                      )
    pose_40=np.array([  [6.8759292e-001	 ,-6.5657324e-001	,  3.0922931e-001,	 -3.3450046e-001,	],
  [5.3356105e-001	,  7.4613512e-001,	  3.9764526e-001	, -2.5476921e-001]	,
 [-4.9191457e-001	, -1.0841578e-001,	  8.6355138e-001,	  3.9842117e-001],
  [0.0000000e+000	,  0.0000000e+000	,  0.0000000e+000,	  1.0000000e+000]])
    print(np.linalg.inv(pose_0))
    pcdBrt = pcdB.transform(pose_40)
    pcdArt = pcdA.transform(pose_0)
    finall_pcd =  pcdArt+pcdBrt
    o3d.visualization.draw_geometries([finall_pcd],
                                      window_name="final",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)
if __name__ == "__main__":
  livingroom2_882()
  #   rgbd_frames()
  # scannet0002()
  # livingroom2_2350()
  # head()
