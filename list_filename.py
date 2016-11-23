# coding:utf-8
import numpy as np
import os

filename=[]
path="face/"#faceまでのpath
folder = os.listdir(path)
for j in range(0,len(folder)):
    files = os.listdir(path+folder[j]+"/")
    cla=str(j)
    for i in range(1,200):#len(files)
        filenam=path+folder[j]+"/"+files[i]+" "+cla
        filename.append(filenam)

data=np.array(filename)
np.random.shuffle(data)
length=len(data)
#data=np.split(data,[np.round(length*0.9),length])
#np.savetxt("train.txt", np.array(data[0]),delimiter=",",fmt="%s")
#np.savetxt("test.txt", np.array(data[1]),delimiter=",",fmt="%s")
np.savetxt("text_npy/test.txt", np.array(data),delimiter=",",fmt="%s")
np.savetxt("text_npy/train.txt", np.array(data),delimiter=",",fmt="%s")
np.save("text_npy/face_class", np.array(folder[1:len(folder)]))

#filename=pd.DataFrame(filename)

#filename.to_csv("filename.csv")

#import csv

#with open('some.csv', 'w') as f:
#  writer = csv.writer(f, lineterminator=',') # 改行コード（\n）を指定しておく
#writer.writerow(filename) # 2次元配列も書き込める
