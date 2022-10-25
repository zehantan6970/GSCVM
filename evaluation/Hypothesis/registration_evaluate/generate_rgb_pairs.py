from frozenDir import relativePath
# ----------------------------------------------------------------------------------
# 生成和gt.log相同的pair对,提供给superglue进行粗配准，superglue生成粗配准的图片对、text3d、text2d
# ----------------------------------------------------------------------------------
# gt_log=relativePath()+'/registration_evaluate/scannet_eval/scene0002_00/scene0002_00_evaluation/scene0002_00_fragments/scene0002_00-evaluation/lowgt.log'
# pairs_txt=relativePath()+'/registration_evaluate/scannet_eval/scene0002_00/scene0002_00_evaluation/scene0002_00_fragments/scene0002_00-evaluation/pairs.txt'
# imageTail=".jpg"
# with open(pairs_txt,mode="w") as p:
#     with open(gt_log,mode="r") as f:
#         lines=f.readlines()
#         for i ,line in enumerate(lines):
#             if not i%5:
#                 id_lst=list(map(int,line.strip().split(" ")))
#                 id1,id2=id_lst[0],id_lst[1]
#                 p.write(str(id1)+imageTail+" "+str(id2)+imageTail+"\n")
# ----------------------------------------------------------------------------------
# 生成全局pair对,提供给superglue进行粗配准，superglue生成粗配准的图片对、text3d、text2d
# ----------------------------------------------------------------------------------
pairs_txt=relativePath()+'/registration_evaluate/heads_eval/heads/heads_evaluation/heads_fragments/heads-evaluation/pairs.txt'
imageTail=".png"
fragmentsNum=53
with open(pairs_txt,mode="w") as p:
    for i in range(fragmentsNum):
        for j in range(i+1,fragmentsNum):
            p.write(str(i)+imageTail+" "+str(j)+imageTail+"\n")

