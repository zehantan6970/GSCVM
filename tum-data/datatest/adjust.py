import os
import cv2
def deletenodepth(path):
    path = '/home/wzh/supergan/tum-data/datatest/'
    with open(path+'associate.txt', 'r') as f:
        for l in f.readlines():
            items = l.split()
            rgb = path + items[1]
            depth = path + items[3]
            rgbimg = cv2.imread(rgb)
            depthimg = cv2.imread(depth, -1)
            rgbimg[depthimg == 0] = 0

        #rgb_filt = rgbimg.copy()
        #gray = cv2.cvtColor(rgb_filt, cv2.COLOR_BGR2GRAY)

            cv2.imshow('img', rgbimg)
            cv2.waitKey(1000)
            cv2.imwrite('/home/wzh/supergan/tum-data/datatest/rgb/' + items[0] + '.png', rgbimg)





