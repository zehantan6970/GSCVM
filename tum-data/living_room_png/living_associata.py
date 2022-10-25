import os
if not os.path.exists('./output'):
    os.makedirs('./output')
outfile = open('./output/associate.txt', 'w')
with open('/home/wzh/supergan/tum-data/living_room_png/associations.txt', 'r') as f:
    for l in f.readlines():
        itemss = l.split()
        outfile.write(itemss[2] +' '+itemss[3]+' '+itemss[0]+itemss[1]+ '\n')
f.close()
outfile.close()