import os
import cv2
path = './'
filelist = os.listdir(path)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
videoWriter = cv2.VideoWriter('output.avi',fourcc, 20, (3840,2160),True)

for i in range(0,519):
    fileName = 'frame'+str(i)+'.jpg'
    img = cv2.imread(fileName)
    videoWriter.write(img)