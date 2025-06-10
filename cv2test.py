import cv2
import numpy as np

cv2.namedWindow("Test")
img = np.full((300, 400, 3), (40, 200, 255), dtype=np.uint8)
cv2.imshow("Test", img)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()