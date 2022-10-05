import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_blue = np.array([0,100,100])
	upper_blue = np.array([5,255,255])
	blur = cv2.GaussianBlur(frame, (35,35), 0)
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	#res = cv2.bitwise_and(frame,frame, mask= mask)
	x,y,w,h = cv2.boundingRect(mask)
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	#cv2.imshow('res', res)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
