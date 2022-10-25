import numpy
import cv2
from PIL import Image
import os

def calculate(image1, image2):
    image1 = cv2.cvtColor(numpy.asarray(image1), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(numpy.asarray(image2), cv2.COLOR_RGB2BGR)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    print(degree)
    return degree


def lvingRoom(txtpath):
    rootpath = txtpath
    txtpathlist = os.listdir(rootpath)
    txtpathlist=sorted(txtpathlist,key=lambda x:int(x[:-4]))
    print(txtpathlist)
    pairs=[]
    with open("pairs.txt", mode="w") as f:
        f.close()
    for txtpath1 in txtpathlist[:400]:
        for txtpath2 in txtpathlist[:400]:
            print(txtpath2)
            clip = numpy.random.random_integers(300, 400)
            if (int(txtpath2[:-4])-int(txtpath1[:-4]))>clip:
                img1=rootpath+txtpath1
                img2=rootpath+txtpath2
                img1 = Image.open(img1)
                img2 = Image.open(img2)
                overlapdegree=calculate(img1,img2)
                print(overlapdegree)
                if 0.25<overlapdegree[0]<0.4:
                    pairs.append([txtpath1,txtpath2])
                    with open("pairs.txt", mode="a") as f:
                        f.write((txtpath1+" "+txtpath2+"\n"))
                        f.close()
                    break
# lvingRoom("F:\\Gree\\Data\\living_room_png\\rgb\\")
img1="F:\\Gree\\Data\\living_room_png\\rgb\\7.png"
img2="F:\\Gree\\Data\\living_room_png\\rgb\\338.png"
img1=cv2.imread(img1)
img2=cv2.imread(img2)
calculate(img1,img2)