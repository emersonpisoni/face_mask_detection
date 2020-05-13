from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

path = "C://Users//Emerson//Desktop//"

detector = MTCNN()
model = load_model(path + "detector.h5")
size = (160, 160)

cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  
  faces = detector.detect_faces(frame)
  
  for face in faces:
    x1, y1, w, h = face['box']
    
    x2 = x1 + w
    y2 = y1 + h
    
    roi = frame[y1: y2, x1:x2]
    
    if np.sum([roi]) !=0:
      roi = cv2.resize(roi, size) ## redimensionamento
      roi = (roi.astype('float')/255.0) ## normalização
      pred = model.predict([[roi]]) ## predição
      
      color = (255, 255, 255)
      
      pred = pred[0]
      
      if pred[0] >= pred[1]:
        label = 'NO MASK'
        color = (0, 0, 255)
      else:
        label = 'MASK'
        color = (0, 255, 0)
      
      label_position = (x1, y1)

      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
      cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)

  cv2.imshow("EMERSHOW", frame)
  
  key = cv2.waitKey(1)
  if key == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()