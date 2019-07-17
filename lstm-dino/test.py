from keras.models import load_model
import numpy as np
from PIL import ImageGrab
import cv2
import cv2 as cv
import keyboard
import time
start = time.time()

model = load_model('dino.h5')

template1 = cv.imread('images/cactus1.png', 0)
template2 = cv.imread('images/cactus2.png', 0)
template3 = cv.imread('images/bird1.png', 0)
template4 = cv.imread('images/bird2.png', 0)
w1, h1 = template1.shape[::-1]
w2, h2 = template2.shape[::-1]
w3, h3 = template3.shape[::-1]
w4, h4 = template4.shape[::-1]

 
def process_img(original_image):
    img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    return img_gray

while(True):
    if keyboard.is_pressed('left'):
        start = time.time()
    data = [0, 0, 0, 0, 0]
    Input = np.array([0, 0, 0, 0, 0])

    screen =  np.array(ImageGrab.grab(bbox=(1126, 267, 1756, 431)))
    score_screen =  ImageGrab.grab(bbox=(1678, 286, 1732, 301))
    img_gray = process_img(screen)
 
    res1 = cv.matchTemplate(img_gray, template1, cv.TM_CCOEFF_NORMED)
    res2 = cv.matchTemplate(img_gray, template2, cv.TM_CCOEFF_NORMED)
    res3 = cv.matchTemplate(img_gray, template3, cv.TM_CCOEFF_NORMED)
    res4 = cv.matchTemplate(img_gray, template4, cv.TM_CCOEFF_NORMED)
    
    threshold = 0.8
    loc1 = np.where(res1 >= threshold)
    loc2 = np.where(res2 >= threshold)
    loc3 = np.where(res3 >= threshold)
    loc4 = np.where(res4 >= threshold)

    if keyboard.is_pressed('left'):
        start = time.time()
 
    for pt in zip(*loc1[::-1]):
        cv.rectangle(img_gray, pt, (pt[0] + w1, pt[1] + h1), (0, 0, 255), 2)
        data[1] = pt[0]
        break
        
    for pt in zip(*loc2[::-1]):
        cv.rectangle(img_gray, pt, (pt[0] + w2, pt[1] + h2), (0, 0, 255), 2)
        data[2] = pt[0]
        break
        
    for pt in zip(*loc3[::-1]):
        cv.rectangle(img_gray, pt, (pt[0] + w3, pt[1] + h3), (0, 0, 255), 2)
        data[3] = pt[0]
        data[4] = pt[1]
        break
        
    for pt in zip(*loc4[::-1]):
        cv.rectangle(img_gray, pt, (pt[0] + w4, pt[1] + h4), (0, 0, 255), 2)
        data[3] = pt[0]
        data[4] = pt[1]
        break

    data[0] = time.time() - start
    Input[0] = data[0]
    Input[1] = data[1]
    Input[2] = data[2]
    Input[3] = data[3]
    Input[4] = data[4]
    Input = np.expand_dims(Input, axis=0)
    Input = np.reshape(Input, (Input.shape[0], 1, Input.shape[1]))
    result = model.predict_classes(Input)
    
    print(Input)
    print(result)
    if (result == 1):
        keyboard.press('space')
 
    cv2.imshow('window', img_gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        f.close()
        break     

