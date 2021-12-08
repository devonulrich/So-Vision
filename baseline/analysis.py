import cv2
import math
import pickle
from evaluate import ANIMOO
def main():
    objects = []
    with (open("anime_mtcnn.pkl", "rb")) as openfile:
        while True:
            try:
                objects=pickle.load(openfile)
            except EOFError:
                break
    print(len(objects))
    metrix = [0,0,0]
    truepos = []
    falsepos = []
    neg = []
    for i in range(len(objects)) :
        img  = objects[i]
        temp = img.compute_metrics()
        if(temp[0]>0):
            truepos.append(img)
        if(temp[1]>0):
            falsepos.append(img)
        if(temp[2]>0):
            neg.append(img)
        metrix[0]+= temp[0]
        metrix[1] += temp[1]
        metrix[2] += temp[2]
    print(metrix)
    print(metrix[0]/sum(metrix))

    for i in range(len(falsepos)):
        img = falsepos[i]
        img.draw()
    


if __name__ == '__main__':
    main()