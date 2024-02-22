import cv2

vid = cv2.VideoCapture("body-detection/body.mp4")

body_cascade = cv2.CascadeClassifier("body-detection/fullbody.xml")

while 1:
    ret, frame = vid.read()
    
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bodies = body_cascade.detectMultiScale(gray, 1.3, 2)
    
    for x,y,w,h in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,25,0), 2)
        
    cv2.imshow("frame", frame)
    cv2.waitKey(20)
    
vid.release()
cv2.destroyAllWindows()