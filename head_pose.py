import sys
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from numpy.lib.type_check import imag


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(sys.argv[1])

image = cv2.imread(sys.argv[2])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

size = image.shape
shape = None

for rect in rects:
    shape = predictor(gray, rect)
    shape = np.array(face_utils.shape_to_np(shape))


### Required features for head pose estimation ###
# Tip of the nose
# Chin
# Left corner of the left eye
# Right corner of the right eye
# Left corner of the mouth
# Right corner of the mouth
image_points = np.array(
    [
        (shape[33, :]),     # Nose tip
        (shape[8,  :]),     # Chin
        (shape[36, :]),     # Left eye left corner
        (shape[45, :]),     # Right eye right corne
        (shape[48, :]),     # Left Mouth corner
        (shape[54, :])      # Right mouth corner
    ],
    dtype="double"
)
 
# Standard 3D model points.
model_points = np.array(
    [
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner                     
    ]
)

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
    [
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ],
    dtype = "double"                     
)
print ("Camera Matrix :\n {0}".format(camera_matrix))


dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
print ("Rotation Vector:\n {0}".format(rotation_vector))
print ("Translation Vector:\n {0}".format(translation_vector))
 
# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose
(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 
for p in image_points:
    cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
cv2.line(image, p1, p2, (0,255,255), 2) 
cv2.imshow("Output", image)
cv2.waitKey(10000)
