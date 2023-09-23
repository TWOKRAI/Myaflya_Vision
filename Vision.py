import cv2
import numpy as np
import socket


if __name__ == '__main__':
   def callback(*arg):
      print (arg)

def pass_func(x):
   pass

cv2.namedWindow('frame')
cv2.namedWindow('setting')

cap = cv2.VideoCapture('Vision_1.mp4')
#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

cv2.createTrackbar('HL', 'setting', 11, 255, pass_func)
cv2.createTrackbar('HM', 'setting', 30, 255, pass_func)
cv2.createTrackbar('SL', 'setting', 66, 255, pass_func)
cv2.createTrackbar('SM', 'setting', 255, 255, pass_func)
cv2.createTrackbar('VL', 'setting', 173, 255, pass_func)
cv2.createTrackbar('VM', 'setting', 255, 255, pass_func)

while True:
    hl = cv2.getTrackbarPos('HL','setting')
    hm = cv2.getTrackbarPos('HM','setting')
    sl = cv2.getTrackbarPos('SL','setting')
    sm = cv2.getTrackbarPos('SM','setting')
    vl = cv2.getTrackbarPos('VL','setting')
    vm = cv2.getTrackbarPos('VM','setting')
    
    hsv_min = np.array((hl, sl, vl), np.uint8)
    hsv_max = np.array((hm, sm, vm), np.uint8)
    
    ret, frame = cap.read()

    height_frame = frame.shape[0]
    weight_frame = frame.shape[1]

    cropped = frame[200:800, 0:1920]
    #frame = cv2.bilateralFilter(frame, 9, 75, 75)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    res = cv2.bitwise_and(cropped, cropped, mask = mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    