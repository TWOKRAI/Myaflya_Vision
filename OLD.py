mport cv2
import numpy as np
import socket
from keras import models
import tensorflow as tf

if __name__ == '__main__':
   def callback(*arg):
      print (arg)

def nothing(x):
   pass


buffer1 = []
buffer2 = [] 
line = 15
i = 0
n = 1

robot_xmin = 318
robot_xmax = 128
robot_ymin = -391
robot_ymax = -145

kernel=np.ones((5,5))

cv2.namedWindow( "frame" )
cv2.namedWindow( "track" )

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

model = tf.keras.saving.load_model("myaphly_model_local.keras")

s = socket.socket()
s.bind(('192.168.1.10', 502))
s.listen(1)
conn, addr = s.accept()


cv2.createTrackbar("Area",  'frame', 3000, 50000, nothing)
cv2.createTrackbar("Square",  'frame' , 17000, 100000, nothing)
cv2.createTrackbar("Delta",  'frame' , 240, 500, nothing)
cv2.createTrackbar("TI", 'frame', 255, 255, nothing) 
cv2.createTrackbar("T2", 'frame', 255, 255, nothing)
cv2.createTrackbar("Delta_X",'frame', 6000, 10000, nothing)
cv2.createTrackbar('X', 'frame', 210, 640, nothing)

cv2.createTrackbar('HL', 'track', 0, 255, nothing)
cv2.createTrackbar('SL', 'track', 23, 255, nothing)
cv2.createTrackbar('VL', 'track', 86, 255, nothing)
cv2.createTrackbar('HM', 'track', 26, 255, nothing)
cv2.createTrackbar('SM', 'track', 255, 255, nothing)
cv2.createTrackbar('VM', 'track', 255, 255, nothing)


while True:
   area_c = cv2.getTrackbarPos("Area", 'frame')
   square = cv2.getTrackbarPos('Square','frame')
   delta = cv2.getTrackbarPos('Delta','frame')
   thresh1 = cv2.getTrackbarPos("TI", 'frame') 
   thresh2 = cv2.getTrackbarPos("T2", 'frame')
   delta_X = cv2.getTrackbarPos('Delta_X', 'frame')
   xdetect = cv2.getTrackbarPos('X', 'frame')

   hl = cv2.getTrackbarPos('HL','track')
   sl = cv2.getTrackbarPos('SL','track')
   vl = cv2.getTrackbarPos('VL','track')
   hm = cv2.getTrackbarPos('HM','track')
   sm = cv2.getTrackbarPos('SM','track')
   vm = cv2.getTrackbarPos('VM','track')

   hsv_min = np.array((hl, sl, vl), np.uint8)
   hsv_max = np.array((hm, sm, vm), np.uint8)
   
   ret, frame = cap.read()
   ret2,frameClear = cap.read()

   height_frame = frame.shape[0]
   weight_frame = frame.shape[1]

   cv2.line(frame, (xdetect - line, 0), (xdetect - line, weight_frame), (255, 0, 0), 1)
   cv2.line(frame, (xdetect, 0), (xdetect, weight_frame), (255, 0, 0), 3)
   cv2.circle(frame, (int(weight_frame/2), int(height_frame/2)), 10, (255, 255, 255), - 1)
   
   frame2 = cv2.bilateralFilter(frame, 9, 75, 75)
   hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
   mask = cv2.inRange(hsv, hsv_min, hsv_max)
   res = cv2.bitwise_and(frame2, frame2, mask = mask)
   gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
   canny = cv2.Canny(gray, thresh1, thresh2)
   dil = cv2.dilate(canny, kernel, iterations = 1)

   contours, h = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

   for contour in contours:
      area = cv2.contourArea (contour)
      if area >  float(area_c):
         cv2.drawContours (frame, contour, -1, (200, 200, 0), 3)
         p = cv2.arcLength(contour, True)
         num = cv2.approxPolyDP(contour, 0.03*p, True)
         
         x, y, w, h = cv2.boundingRect(num)

         i = i + 1
         
         if w * h <= square:
            if  -(delta/10) <= (w - h) <= delta/10 or -(delta/10) <= (h - w) <= delta/10:
            
               dx = x + w/2
               dy = y + h/2
        
               cv2.circle(frame, (int(dx), int(dy)), 10, (0, 255, 0), - 1)
               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

               text1 = 'Object: ' + str(i)
               text2 = 'Coordinates: ' + str(dx) + ', ' + str(dy)
            
               cv2.putText(frame, text1, (x, y - 30), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
               cv2.putText(frame, text2, (x, y - 10), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
               
               dx_transp = robot_xmin - int((robot_xmax - robot_xmin)/403*dx)
               dy_transp = robot_ymin + int((robot_ymax - robot_ymin)/height_frame*dy)

               if xdetect - line <= dx <= xdetect:
                  wcropp, hcropp = 128, 128
                  cropp = frameClear[int(dy - hcropp/2): int(dy + hcropp/2), int(dx - wcropp/2): int(dx + wcropp/2)]
            
                  #n = n + 1
                  #cv2.imwrite(f'C:/photo/{n}0.png', cropp)
                  cropp = cv2.cvtColor(cropp, cv2.COLOR_BGR2GRAY)
                  cropp = np.expand_dims(cropp, axis = 0)        
                  predict = model.predict(cropp)
                  print(predict)
                  predict = predict.reshape(2)
            
                  if predict[0] >= 0.7:           
                     buffer1.insert(i - 1, dy_transp)
                
                  
   i = 0

   buffer2 = buffer2 + buffer1
   buffer1.clear()

   xdetect = float(xdetect)

   #print(buffer2)
   
   if len(buffer2)!= 0:
      buff = buffer2.pop(0)
      conn.send(str.encode(f'{delta_X}') + b',' + str.encode(f'{1}') + b',' + str.encode(f'{len(buffer2}') + b',' + str.encode(f'{dy_transp}') + b'\r\n')
      print(str.encode(f'{delta_X}') + b',' + str.encode(f'{1}') + b',' + str.encode(f'{dy_transp}') + b'\r\n')
   #else: 
      conn.send(str.encode(f'{delta_X}') + b',' + str.encode(f'{0}') + b'\r\n')
      print(str.encode(f'{delta_X}') + b',' + str.encode(f'{0}') + b'\r\n')

   if len(buffer2) >= 10:
      conn.send(str.encode(f'{2}') + b'\r\n')
      print(str.encode(f'{delta_X}') + b',' + str.encode(f'{2}') + b'\r\n')
      buffer2.clear()

   cv2.imshow("frame", frame)
   cv2.imshow("track", res)
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
cap.release()
cv2.destroyAllWindows()
