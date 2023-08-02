import cv2
import numpy as np

cap = cv2.VideoCapture(1)

def detect_colors(frame):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([40, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    blue_lower = np.array([90, 100, 100], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    green_lower = np.array([40, 100, 100], np.uint8)
    green_upper = np.array([80, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 30, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    colors = [(yellow_mask, (0, 255, 255), "yellow"),
              (red_mask, (0, 0, 255), "red"),
              (blue_mask, (255, 50, 50), "blue"),
              (green_mask, (0, 255, 0), "green"),
              (white_mask, (255, 255, 255), "white"),
              (black_mask, (0, 0, 0), "black")]

    for color_mask, color, name in colors:
        contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                text_x = x + (w - text_size[0]) // 2
                text_y = y + h + text_size[1] + 10
                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                break

    return frame

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frame = detect_colors(frame)
    
    cv2.imshow("Color Detection", frame)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()