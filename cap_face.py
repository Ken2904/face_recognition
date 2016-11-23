# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
color = (0, 187, 254) 
cap = cv2.VideoCapture(0)
n=300#データ数
#cascade_path = "/Users/kenichirosakaba/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
#cascade_path = "/Users/kenichirosakaba/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
#cascade_path = "/Users/kenichirosakaba/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
cascade_path = "/Users/kenichirosakaba/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
#cascade_path = "/Users/kenichirosakaba/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#cascade_path = "/Users/kenichirosakaba/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml"
cascade = cv2.CascadeClassifier(cascade_path)
dir_path="face" 
#os.mkdir(dir_path)
print("誰の顔データを取りますか？")
person=raw_input("")
ind_path=dir_path + "/" + person
os.mkdir(ind_path)
loop=0
i=0
image=range(400)
while True:
	loop+=1
	print("running")
	ret, frame = cap.read()
	frame = cv2.resize(frame, (720,480))
	frame=cv2.flip(frame,1)
	#frame = frame[:,::-1]
	frame_gray=cv2.cvtColor(frame,cv2.cv.CV_BGR2GRAY)
    	facerect = cascade.detectMultiScale(frame_gray, rejectLevels=[],levelWeights=[],scaleFactor=1.5, minNeighbors=1 , minSize=(10, 10)) 
    	for (x,y,w,h) in facerect:
        	cv2.rectangle(frame, (x,y),(x+w,y+h), color, thickness=1)
        	print((loop%10==0 and i <=n),i,loop)
        	if (loop%1==0 and i <=n):
        		dst = frame[y:y+h, x:x+w]
        		dst = cv2.resize(dst, (100,100))#(幅，高さ，チャンネル)
        		new_image_path = ind_path + '/' + str(i) +person+ ".jpg"
			cv2.imwrite(new_image_path, dst)
			print("saved photo")
			image[i]=dst#(高さ，幅，チャンネル)
			i += 1
	cv2.imshow('camera capture', frame)
	print("Runnig")
	cv2.waitKey(5)
	k = cv2.waitKey(1) # 1msec待つ
        if k == 27 or i>=n: # ESCキーで終了
        	print("データを取り終わりました．")
        	break
#data=np.zeros((n,3,48,72))#(ミニバッチ数．高さ，幅，チャンネル)

#for i in range(n):
#	data[i]=np.transpose(image[i],axes=(2,0,1))
#data_name="/Users/kenichirosakaba/Desktop/face/numpy_data"+"/"+ person
#np.save(data_name,data)
cap.release()
cv2.destroyAllWindows()