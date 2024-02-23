import numpy as np 
import cv2
import os 
data_path = './data/'
faceData = []
labels = [] 
names= {}
offset = 20
classId = 0
for f in os.listdir(data_path):
    if f.endswith('.npy'):
        # X 
        names[classId] = f[:-4]
        dataItem = np.load(data_path + f)
        faceData.append(dataItem)
        m = dataItem.shape[0]
        #target 
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)
X = np.concatenate(faceData, axis=0)
yt = np.concatenate(labels, axis=0).reshape(1, -1)
print(yt.shape)
#algorithm 
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance      
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = KNeighborsClassifier(
                  n_neighbors=1,
                  n_jobs=1, 
                  metric = distance.correlation)

classifier.fit(X, yt)


cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    success, img = cam.read() 
    if not success:
        raise "can not read "
    faces = model.detectMultiScale(img, 1.3, 1)
    for f in faces:

        f = faces[-1]
        x,y,w,h = f
        #crop and save the largest face 
        cropp_face = img[y - offset:y+h + offset, x - offset:x+w + offset]

        cropp_face = cv2.resize(cropp_face, (200, 200))
     
        cv2.imshow("Image show ", img)

        classPredicted = classifier.predict(cropp_face.reshape(1,-1))
        namePredicted = names[classPredicted[0]]
        cv2.putText(img, namePredicted, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1 , (255,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,200,100),2)
    cv2.imshow("predicted image", img)
    exit = cv2.waitKey(1)
    if exit == ord('q'):
        break 

cam.release()
cv2.destroyAllWindows()