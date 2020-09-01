import cv2
import numpy as np
bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)
history = 30
bs.setHistory(history)
frames = 0
camera = cv2.VideoCapture("##.mp4")
count = 0

while True:
    ret, frame = camera.read()
    if ret == True:
        fgmask = bs.apply(frame)
        if frames < history:
            frames += 1
            continue
        print('Read a new frame: ', ret)

        th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                       iterations=None)
        dilated = cv2.dilate(th,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (3, 3)),
                             iterations=None)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite("frame%d.jpg" % count, frame)

        cv2.imshow("knn", fgmask)
        cv2.imshow("thresh", th)
        cv2.imshow("diff", frame & cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))
        cv2.imshow("detection", frame)
        count += 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
camera.release()
cv2.destroyAllWindows()