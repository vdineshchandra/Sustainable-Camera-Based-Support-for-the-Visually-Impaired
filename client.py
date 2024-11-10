from datetime import datetime
import cv2
import requests

# Raspberry Pi camera parameters
WIDTH = 640
HEIGHT = 480

# Laptop server URL
SERVER_IP = "192.168.1.129"
SERVER_URL = "http://" + SERVER_IP + ":5000/predict"

# Raspberry Pi camera initialization
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
# cap.set(cv2.CAP_PROP_FPS, FPS)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Resize frame if needed
    # frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Convert frame to JPEG format
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_data = img_encoded.tobytes()

    # Send frame to laptop server for prediction
    files = {'image': image_data}
    response = requests.post(SERVER_URL, files=files)
    prediction = ""
    confidence = 0

    # Process prediction response
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        confidence = result['confidence']

        # Do something with prediction and confidence
        print(f'Prediction: {prediction}, Confidence: {confidence}')
    else:
        print(f'Error: {response.text}')

    if prediction == "Human" and confidence > 0.8:
        frame = cv2.putText(frame, str(datetime.now()), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        frame = cv2.putText(frame, "Human"+" "+str(confidence), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
    if prediction == "Vehicle" and confidence > 0.8:
        frame = cv2.putText(frame, str(datetime.now()), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        frame = cv2.putText(frame, "Vehicle"+" "+str(confidence), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
    if prediction == "Pet" and confidence > 0.8:
        frame = cv2.putText(frame, str(datetime.now()), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        frame = cv2.putText(frame, "Pet", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
    if prediction == "Traffic" and confidence > 0.8:
        frame = cv2.putText(frame, str(datetime.now()), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        frame = cv2.putText(frame, "Traffic", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
        
    

    # Display frame if needed
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
