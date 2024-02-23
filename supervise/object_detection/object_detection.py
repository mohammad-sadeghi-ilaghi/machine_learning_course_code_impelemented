import cv2
import numpy as np 
#Create a Camera Object  
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
name = input("what is you name: ")
offset = 20

#Create a list of face and save 
faceData = []
skip =0
# Read image from  Camera Object 
while True:
    success, img = cam.read() 
    if not success:
        raise "can not read "
    #store the gray image 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(img, 1.3, 5)
    #store to pick the bigest image 
    faces = sorted(faces, key=lambda f: f[2]*f[3])
    if len(faces) > 0:
        f = faces[-1]
        x,y,w,h = f
        #crop and save the largest face 
        cropp_face = img[y - offset:y+h + offset, x - offset:x+w + offset]
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,200,100),2)
        cropp_face = cv2.resize(cropp_face, (200, 200))
        skip+=1
    if len(faceData) % 10 == 0 :
        faceData.append(cropp_face)        
    cv2.imshow("Image show ", img)


    exit = cv2.waitKey(1)
    if exit == ord('q') or skip > 10:
        break 

faceData = np.asarray(faceData)
m = faceData.shape[0]
faceData = faceData.reshape((m, -1))
np.save(f"./data/{name}.npy", faceData)

cam.release()
cv2.destroyAllWindows()
print("Successfully saves")