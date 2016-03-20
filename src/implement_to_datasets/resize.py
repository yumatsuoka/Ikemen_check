import cv2
import sys

#import image
img = cv2.imread(sys.argv[1])

#change size
selected_size = 128
resized_img = cv2.resize(img, (selected_size, selected_size))

#save image
cv2.imwrite("resized"+sys.argv[1], resized_img)
