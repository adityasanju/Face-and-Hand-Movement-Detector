import cv2
import HandTrackingModule as htm
import numpy as np

caffe_model = 'weights.caffemodel'
prototxt_file = 'deploy.prototxt.txt'
min_confidence = 0.2


net = cv2.dnn.readNetFromCaffe(prototxt_file,caffe_model)

cap = cv2.VideoCapture(0)
hands = htm.handDetector()

while True:
    dummy, image = cap.read()
    dummy2, img = cap.read()
    img = hands.findHands(img)
    h , w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    face_found = net.forward()

    for i in range(face_found.shape[2]):
        confidence = face_found[0][0][i][2]
        if confidence > min_confidence:
            upper_left_x = int(face_found[0, 0, i, 3] * w)
            upper_left_y = int(face_found[0, 0, i, 4] * h)
            lower_right_x = int(face_found[0, 0, i, 5] * w)
            lower_right_y = int(face_found[0, 0, i, 6] * h)
            confidence = confidence*100
            text = f'{confidence:.3f}%'

            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 255, 0))
            cv2.putText(image , text, (upper_left_x,upper_left_y),cv2.FONT_HERSHEY_DUPLEX, 1,  (0, 255, 0) , 2)

    cv2.imshow("detected faces", image)
    cv2.imshow("hands", img)
    cv2.waitKey(5)
cv2.destroyAllWindows()
cv2.release()
