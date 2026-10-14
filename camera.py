import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

while True:

    ret, frame = camera.read()

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1)==27:
        break

camera.release()
cv2.destroyAllWindows()