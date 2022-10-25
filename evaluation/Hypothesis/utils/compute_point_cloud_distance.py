import open3d as o3d
import numpy as np


print("->正在加载点云1... ")
# pcd1 = o3d.io.read_point_cloud("F:\Gree\Data\living_room_png\ply\\587.ply")5.png
pcd1 = o3d.io.read_point_cloud("F:\Gree\Data\living_room_png\ply\\136.ply")
print(pcd1)
print("->正在加载点云2...")
# pcd2 = o3d.io.read_point_cloud("F:\Gree\Data\living_room_png\ply\\380.ply")#359.png
pcd2 = o3d.io.read_point_cloud("F:\Gree\Data\living_room_png\ply\\859.ply")
print(pcd2)

print("->正在点云1每一点到点云2的最近距离...")
dists = pcd1.compute_point_cloud_distance(pcd2)
dists = np.asarray(dists)
print("->正在打印前10个点...")
print(dists[:10])

print("->正在提取距离小于1.56的点")
ind = np.where(dists < 1.56)[0]
pcd3 = pcd1.select_by_index(ind)
print(np.asarray(pcd3.points))
# pcd3.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pcd3], window_name="计算点云距离",
                                  width=800,  # 窗口宽度
                                  height=600)  # 窗口高度
