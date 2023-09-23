import cv2
import numpy as np
import socket


if __name__ == '__main__':
   def callback(*arg):
      print (arg)

def pass_func(x):
   pass

cv2.namedWindow('frame')

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

cv2.createTrackbar('HL', 'frame', 11, 255, pass_func)
cv2.createTrackbar('SL', 'frame', 73, 255, pass_func)
cv2.createTrackbar('VL', 'frame', 173, 255, pass_func)
cv2.createTrackbar('HM', 'frame', 23, 255, pass_func)
cv2.createTrackbar('SM', 'frame', 255, 255, pass_func)
cv2.createTrackbar('VM', 'frame', 255, 255, pass_func)

while True:
    hl = cv2.getTrackbarPos('HL','frame')
    sl = cv2.getTrackbarPos('SL','frame')
    vl = cv2.getTrackbarPos('VL','frame')
    hm = cv2.getTrackbarPos('HM','frame')
    sm = cv2.getTrackbarPos('SM','frame')
    vm = cv2.getTrackbarPos('VM','frame')
    
    hsv_min = np.array((hl, sl, vl), np.uint8)
    hsv_max = np.array((hm, sm, vm), np.uint8)
    
    ret, frame = cap.read()

    #frame = cv2.bilateralFilter(frame, 9, 75, 75)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    cv2.imshow("frame", res)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    