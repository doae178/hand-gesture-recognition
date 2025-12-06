import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('hand_model.h5')


labels = ['fist', 'ok', 'palm', 'peace']


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    img = cv2.resize(frame, (224, 224))            
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    img = img.astype("float32") / 255.0          
    img = np.expand_dims(img, axis=0)              

    prediction = model.predict(img)
    gesture = labels[np.argmax(prediction)]

    cv2.putText(frame, gesture, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
