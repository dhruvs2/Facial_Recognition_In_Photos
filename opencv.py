import cv2,time

face_cascade = cv2.CascadeClassifier("/home/dhruv/haarcascade_frontalface_default.xml") # All facial information predefined in the file

img = cv2.imread("/home/dhruv/Europa.png",1)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to grayscale for easier reading
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, 
                                      minNeighbors = 5)
# Making boundary on the Face
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)

resized = cv2.resize(img, (int(img.shape[1]),int(img.shape[0])))
cv2.imshow("Gray",img)
cv2.waitKey(0)

print(type(faces))
print(faces)
