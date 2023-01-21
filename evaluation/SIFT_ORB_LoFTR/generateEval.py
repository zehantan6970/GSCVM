import os
from shutil import copyfile
import cv2 as cv
from frozenDir import relativePath
rootPath=relativePath()+"/registration_evaluate/heads_eval/heads/"
targetPath=relativePath()+"/registration_evaluate/heads_eval/heads/heads_evaluation/"
"""
    对文件批量重新命名并复制到其他文件夹中
    注意图像不仅需要重命名还需要resize
"""
# ----------------------------------------------------------------------------------
# rename pose
# ----------------------------------------------------------------------------------
root_pose_path=rootPath+"pose/"
taregt_pose_path=targetPath+"pose_eval/"
pose_names=os.listdir(root_pose_path)
for i,pose_name in enumerate(pose_names):
    print(pose_name)
    pose_file=root_pose_path+pose_name
    with open(taregt_pose_path+str(i)+".info.txt",mode="w") as f:
        with open(pose_file,mode="r") as r:
            lines=r.readlines()
            for line in lines:
                value=list(map(float,line.strip().split()))
                f.write(str(value[0])+" "+str(value[1])+" "+str(value[2])+" "+str(value[3])+"\n")
            r.close()
        f.close()
# ----------------------------------------------------------------------------------
# rename ply
# ----------------------------------------------------------------------------------
root_ply_path=rootPath+"ply/"
taregt_ply_path=targetPath+"ply_eval/"
ply_names=os.listdir(root_ply_path)
for i,ply_name in enumerate(ply_names):
    print(ply_name)
    ply_file=root_ply_path+ply_name
    copyfile(ply_file,taregt_ply_path+str(i)+".ply")
# ----------------------------------------------------------------------------------
# rename rgb
# ----------------------------------------------------------------------------------
root_color_path=rootPath+"color/"
taregt_color_path=targetPath+"color_eval/"
color_names=os.listdir(root_color_path)
for i,color_name in enumerate(color_names):
    print(color_name)
    color_file=root_color_path+color_name
    img=cv.imread(color_file)
    # ----------------------------------------------------------------------------------
    # rgb和depth的shape一致
    # ----------------------------------------------------------------------------------
    resized_img=cv.resize(img,(640,480))
    cv.imwrite(taregt_color_path+str(i)+".png",resized_img)

# ----------------------------------------------------------------------------------
# rename depth
# ----------------------------------------------------------------------------------
root_depth_path=rootPath+"depth/"
taregt_depth_path=targetPath+"depth_eval/"
depth_names=os.listdir(root_depth_path)
for i,depth_name in enumerate(depth_names):
    print(depth_name)
    depth_file=root_depth_path+depth_name
    copyfile(depth_file,taregt_depth_path+str(i)+".png")
