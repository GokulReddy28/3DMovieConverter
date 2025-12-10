import cv2

cap = cv2.VideoCapture("mybigmovie.mp4")

print("Width:", cap.get(3))
print("Height:", cap.get(4))
print("FPS:", cap.get(5))

cap.release()
