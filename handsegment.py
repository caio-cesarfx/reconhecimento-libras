import numpy as np
import cv2

limites = [
    ([0, 120, 0], [140, 255, 100]),
    ([25, 0, 75], [180, 38, 255])
]

frame = r"C:\Users\Caio Cesar\Desktop\TCC\SharedScreenshot.jpg"


def handsegment(frame):
    lower, upper = limites[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask1 = cv2.inRange(frame, lower, upper)

    lower, upper = limites[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper)

    mask = cv2.bitwise_or(mask1, mask2)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    return output


handsegment(frame)

if __name__ == '__main__':
    frame = cv2.imread("test.jpg")
    handsegment(frame)
