import cv2
#Create a Camera Object  
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Read image from  Camera Object 
while True:
    success, img = cam.read() 
    if not success:
        raise "can not read "
    faces = model.detectMultiScale(img, 1.3, 1)
    for f in faces:
        x,y,w,h = f
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,200,100),2)
    cv2.imshow("Image show ", img)

    exit = cv2.waitKey(1)
    if exit == ord('q'):
        break 

cam.release()
cv2.destroyAllWindows()
