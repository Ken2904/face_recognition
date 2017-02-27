#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import os
import math
import NNmodel as NN
import list_filename 


face_class=np.load("text_npy/face_class.npy")

NUM_CLASSES = len(face_class)
IMAGE_SIZE = 96
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3


# サンプル顔認識特徴量ファイル
cascade_path = "...path.../anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml" #人の場合はこっち
#cascade_path = "...path.../anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/lbpcascades/lbpcascade_animeface.xml" # アニメの場合はこっち
 
images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
keep_prob = tf.placeholder("float")
logits = NN.inference(images_placeholder, keep_prob)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess, "model/model.ckpt")
		

while True:
	if __name__ == '__main__':
		color = (0, 187, 254) 
		cap = cv2.VideoCapture(0)
		cascade_path = "...path.../anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
		cascade = cv2.CascadeClassifier(cascade_path)
		print("カメラ〜〜〜〜")
		loop=0
		i=0
		image=range(200)
		points=range(3)
		ii=0
		while True:
			loop+=1
			print("running")
			ret, frame = cap.read()
			frame = cv2.resize(frame, (720,480))
			frame=cv2.flip(frame,1)
			#frame = frame[:,::-1]
			frame_gray=cv2.cvtColor(frame,cv2.cv.CV_BGR2GRAY)
	    		facerect = cascade.detectMultiScale(frame_gray, rejectLevels=[],levelWeights=[],scaleFactor=1.5, minNeighbors=1 , minSize=(10, 10)) 
	    		x=0
	    		for (x,y,w,h) in facerect:
	        		cv2.rectangle(frame, (x,y),(x+w,y+h), color, thickness=2)
			#if x!=0:# and len(facerect)==1:
				#new_point=[x,y]
				#points=[points[1],points[2],new_point]
				#ii+=1
			#点の距離を求める
			#def dist(p1,p2):
				#a=(p1[0]-p2[0])**2
				#b=(p1[1]-p2[1])**2
				#return(math.sqrt(a+b))
			#if ii >3 :
				#total_dist=dist(points[0],points[1])+dist(points[2],points[1])
				#print "tatal dist="+ str(total_dist)
			cv2.imshow('camera capture', frame)
			print("Runnig")
			k=cv2.waitKey(8)
			#if ii>3:
				#if total_dist<5:
				#	break
			if k==27:
				os.system('afplay /Users/kenichirosakaba/Downloads/camera-shutter1.mp3 ' )
				break
	
		cv2.imshow('camera capture', frame)
		cv2.waitKey(0)
		#dst = frame[y:y+h, x:x+w]
		cap.release()
		cv2.destroyAllWindows()
		image=frame
		if len(facerect) > 0:
		 	test_image=[]
	 		for (x,y,w,h) in facerect:
				img = frame[y:y+h, x:x+w]
				img = cv2.resize(img, (96, 96))
				test_image.append(img.flatten().astype(np.float32)/255.0)
		else:
			continue
		
		
		#test_image = np.asarray(test_image)
	
		pred=range(len(facerect))
		pred2=range(len(facerect))
		for i in range(len(test_image)):
			test_image[i] = np.asarray(test_image[i])
			pred[i] = np.argmax(logits.eval(feed_dict={images_placeholder: [test_image[i]],keep_prob: 1.0 })[0])
			pred2[i]=logits.eval(feed_dict={images_placeholder: [test_image[i]],keep_prob: 1.0 })[0][ pred[i] ] *100 #パーセント表示
			print pred[i],pred2[i]
			# 以下画像を表示
			print facerect
			p=facerect[i]
			font_size = np.log2(p[3]*p[3])/20
			text = "This is "+ face_class[pred[i]] + ('%2.2f' % pred2[i]) + "%"	#フォントの指定
			font=cv2.FONT_HERSHEY_SIMPLEX
	    	#font = cv2.FONT_HERSHEY_COMPLEX_SMALL
	    	#font = cv2.FONT_HERSHEY_SIMPLEX
	    	#font = cv2.FONT_HERSHEY_PLAIN
	    	#文字の書き込み
			print str(p)+"aaaaaaaaaaaa"
			cv2.putText(image,text,(p[0],p[1]),font, font_size,(0,0,255),3)
			print [x,y,w,h]
		print face_class
		print np.array(pred)
		cv2.imshow('Result',image) # 出力させる画像のタイトル(変更可)
		tex=""
		for i in range(0,len(pred)):
			if len(pred)==1:
				os.system('say -r 130 '+ 'this is '+ face_class[pred[i]] )
			else:
				tex=""
				for j in range(0,len(pred)):
					tex=tex+" "+face_class[pred[j]]
					if j==len(pred)-1:
						break
					tex=tex+", and ,"
		if len(pred)!=1:			
			os.system("say -r 130 "+ "they are "+ tex )
			print tex
				
			
	        
		while(1):
			if cv2.waitKey(10) > 0:
				break # 出力した画像上でキーボード入力するとプログラムが終了する
				
	print len(test_image)
	print pred

	kk=cv2.waitKey(0)
	print 'esc:again,'
	print 'other: one more'
	if kk == 27:
		break



		