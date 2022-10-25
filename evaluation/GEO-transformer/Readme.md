GeoTransformer installation: https://github.com/qinzheng93/GeoTransformer
ps:Model input point cloud needs to be converted to npy file with dimension N*3
1. Two sets of point cloud stitching (demo.py)
Parameter description --src_file source point cloud file, --ref_file target point cloud file, --gt_file 4*4 transformation matrix, --weights pre-trained model
Run the command line
python demo.py --src_file=... /... /data/demo/src.npy --ref_file=... /... /data/demo/ref.npy --gt_file=... /... /data/demo/gt.npy --weights=... /... /weights/geotransformer-3dmatch.pth.tar
2, scene stitching (rebuild_livingroom.py)
Parameters description --npypath point cloud npy file path, --plypath point cloud npy file path, --weights pre-trained model, --frames
total number of frames in the point cloud, --skip stitched point cloud interval
Run the command line
python rebuild_livingroom.py -npypath=... /... /data/newNpy/ --plypath=... /... /data/newPly/ --weight=... /... /weights/geotransformer-3dmatch.pth.tar -frames=810 --skip=10
3, recall_precision.py is for evaluation to generate log files containing point cloud RT files, Run the command line: python recall_precision.py 