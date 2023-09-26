import cv2
import numpy as np
import socket
import random
from keras import models
import tensorflow as tf


if __name__ == '__main__':
   def callback(*arg):
      print (arg)

buffer = []
buffer_test_in = []
buffer_test_out = []
data = 'photo'
marker = 0
i = 0
n = 0

bad = 0 
find = 0

robot_xmin = 318
robot_xmax = 128
robot_ymin = -391
robot_ymax = -145


def pass_func(x):
   pass

def randomizer(x, y, r, d):
    global i
    global bad
    global t

    global myseq
    global defect
    global deltax 
    global deltay 
    global size 
    global color 
    global line1 
    global line2 
    global rand1
    global rand2
    
    
    list_random = [1] + [0] * d
    t = random.choice(list_random)

    if t == 1:
        myseq = ['O', '|', 'L', 'Q', 'D', 'C']
        defect = random.choice(myseq)
        deltax = random.randint(-8, 8)
        deltay = random.randint(-8, 8)
        size = 1 / (random.randint(2, 4))
        color = random.randint(25, 75)
        line1 = random.randint(2, 5)
        line2 = random.randint(1, 2)
        rand1 = random.randint(0, 1)
        rand2 = random.randint(0, 1)

        if rand1 == 0 and rand2 == 0:
            rand1 = 1
            rand2 = 0

        if rand1 == 1:
            cv2.putText(output_screen, str(defect), (x-deltax, y - deltay), cv2.FONT_ITALIC, size, (color, color, color), line1, cv2.LINE_AA)
            cv2.putText(output_ai, str(defect), (x-deltax, y - deltay), cv2.FONT_ITALIC, size, (color, color, color), line1, cv2.LINE_AA)
        
        if rand2 == 1:
            cv2.circle(output_screen, (x, y), r, (color, color, color), line2)
            cv2.circle(output_ai, (x, y), r, (color, color, color), line2)
    
        cv2.circle(output_screen, (x, y), r + 5, (255, 0, 0), 2)

        bad += 1
    else: 
        cv2.circle(output_screen, (x, y), r + 5, (0, 255, 0), 2)

    i += 1
    cv2.putText(output_screen, str(i), (x-5, y - 10), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(output_screen, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)


cv2.namedWindow('frame')
cv2.namedWindow('setting')
cv2.namedWindow('setting2')

cap = cv2.VideoCapture('Vision_1.mp4')
#cap = cv2.VideoCapture('http://192.168.43.1:4747/video')

cv2.createTrackbar('HL', 'setting', 0, 255, pass_func)
cv2.createTrackbar('HM', 'setting', 61, 255, pass_func)
cv2.createTrackbar('SL', 'setting', 90, 255, pass_func)
cv2.createTrackbar('SM', 'setting', 255, 255, pass_func)
cv2.createTrackbar('VL', 'setting', 0, 255, pass_func)
cv2.createTrackbar('VM', 'setting', 255, 255, pass_func)

cv2.createTrackbar('dp', 'setting2', 350, 1000, pass_func)
cv2.createTrackbar('minDist', 'setting2', 16, 100, pass_func)
cv2.createTrackbar('param1', 'setting2', 10, 500, pass_func)
cv2.createTrackbar('param2', 'setting2', 2, 100, pass_func)
cv2.createTrackbar('minRadius', 'setting2', 18, 100, pass_func)
cv2.createTrackbar('maxRadius', 'setting2', 28, 200, pass_func)
cv2.createTrackbar('photo', 'setting2', 0, 1, pass_func)

model = tf.keras.saving.load_model("myaphly_model_local.keras")

"""s = socket.socket()
s.bind(('192.168.1.10', 502))
s.listen(1)
conn, addr = s.accept()"""

while True:
    hl = cv2.getTrackbarPos('HL','setting')
    hm = cv2.getTrackbarPos('HM','setting')
    sl = cv2.getTrackbarPos('SL','setting')
    sm = cv2.getTrackbarPos('SM','setting')
    vl = cv2.getTrackbarPos('VL','setting')
    vm = cv2.getTrackbarPos('VM','setting')
    
    c_dp = cv2.getTrackbarPos('dp','setting2')
    c_minDist = cv2.getTrackbarPos('minDist','setting2')
    c_param1 = cv2.getTrackbarPos('param1','setting2')
    c_param2 = cv2.getTrackbarPos('param2','setting2')
    c_minRadius = cv2.getTrackbarPos('minRadius','setting2')
    c_maxRadius = cv2.getTrackbarPos('maxRadius','setting2')
    photo = cv2.getTrackbarPos('photo', 'setting2')

    hsv_min = np.array((hl, sl, vl), np.uint8)
    hsv_max = np.array((hm, sm, vm), np.uint8)
    
    ret, frame = cap.read()

    #data = s.recv(1024) 

    if  photo == 0:
        buffer.clear()
        marker = 0
        cv2.imwrite(f'photo/main.png', frame)
        cv2.setTrackbarPos('photo', 'setting2', 1)
        bad = 0
        find =0
        #data = 0

    frame = cv2.imread('photo/main.png')

    cropped = frame[200:800, 0:1920]
    output_ai = cropped.copy()
    output_screen = cropped.copy()

    height_frame = cropped.shape[0]
    weight_frame = cropped.shape[1]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    res = cv2.bitwise_and(cropped, cropped, mask = mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp = (c_dp/100), minDist = c_minDist, param1 = c_param1, 
                            param2 = (c_param2/100), minRadius = c_minRadius, maxRadius = c_maxRadius)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
           
            if  marker == 0:
                randomizer(x, y, r, 4)
                
                wcropp, hcropp = 48, 48
                cropp = output_ai[int(y - hcropp/2): int(y + hcropp/2), int(x - wcropp/2): int(x + wcropp/2)]
                
                cropp_gray = cv2.cvtColor(cropp, cv2.COLOR_BGR2GRAY)
                #n += 1
                #cv2.imwrite(f'Data_bad/{n}.png', cropp_gray)
            
            dx_transp = robot_xmin - int((robot_xmax - robot_xmin)/weight_frame*x)
            dy_transp = robot_ymin + int((robot_ymax - robot_ymin)/height_frame*y)

            cropp = np.expand_dims(cropp_gray, axis = 0)         

            predict = model.predict(cropp)
            print(predict)
            predict = predict.reshape(2)
         
            if predict[0] >= 0.7 and t == 1:         
                buffer.insert(i - 1, [str.encode(f'{dx_transp}') ,str.encode(f'{dy_transp}')])
                
                find += 1
                cv2.circle(output_screen, (x, y), r, (0, 0, 255), 5)
            
            if predict[0] >= 0.7 and t != 1:    
                print('Error {i}')

            if predict[1] >= 0.7 and t == 1:    
                print('Error {i}')

        i = 0

        if  marker == 0:

            buffer.sort(key = lambda x: x[0], reverse = True)
            #print('ОТСОРТИРОВАННЫЙ:') 
            #print(buffer) 
            print('Передача:') 
                        
            if len(buffer) > 0:
                buffer2 = str(sum(buffer, [])).strip('[]')
                #conn.send(str.encode(f'{len(buffer)}') + b',' + str.encode(f'{buffer2}')+ b'\r\n')
                print(str.encode(f'{len(buffer)}') + b', ' + str.encode(buffer2) + b'\r\n')
            else: 
                #conn.send(str.encode(f'{len(buffer)}') + b'\r\n')
                print(str.encode(f'{len(buffer)}') + b'\r\n') 
            
            marker = 1

            cv2.imwrite(f'photo/screen.png', output_screen)

    output_screen = cv2.imread('photo/screen.png')

    cv2.putText(output_screen, 'FIND  ' + str(find), (40, 50), cv2.FONT_ITALIC, 1.5, (255, 255, 255), 1, cv2.LINE_AA)               
    cv2.putText(output_screen, 'BAD  '+ str(bad), (40, 100), cv2.FONT_ITALIC, 1.5, (255, 255, 255), 1, cv2.LINE_AA)   
    
    cv2.imshow("frame", output_screen)
    cv2.imshow("gray", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    