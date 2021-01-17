import sys
import imutils
from imutils import face_utils
import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(sys.argv[1])

image = cv2.imread(sys.argv[2])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	for (x, y) in shape:
		cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(10000)
