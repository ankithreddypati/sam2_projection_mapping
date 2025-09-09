import cv2

def list_cameras(max_tested=10):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

print("Available camera ports:", list_cameras())
