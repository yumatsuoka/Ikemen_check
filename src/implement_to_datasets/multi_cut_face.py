#-------------------------------------------------------------------------------
# Name:        cut_face from image
# Purpose:
# Author:      yuma
# Created:     24/01/2016
#-------------------------------------------------------------------------------
import cv2
import sys

def main():
    #顔検出器をロード
    face_cascade = cv2.CascadeClassifier('/opt/Install-OpenCV/Ubuntu/OpenCV/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #入力画像の読み込み
    #img = cv2.imread('image (195).jpg')
    img = cv2.imread(sys.argv[1])

    #gray scaleヘ変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #顔検出
    faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(40, 40))
    j = 0
    for i in faces:
        max = 0
        maxh = 0
        maxw = 0
        resx = 0
        resy = 0
        maxw = i[2]
        maxh = i[3]
        resx = i[0]
        resy = i[1]
        max= i[2]*i[3] 
        sub = img[resy:resy+maxh, resx:resx+maxw]
        j += 1
        cv2.imwrite("face_"+str(j) +sys.argv[1], sub)
if __name__ == '__main__':
    main()
