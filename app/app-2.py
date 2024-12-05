import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pretrained emotion recognition model
pretrained_model = load_model('../saved_models/emotion_recognition_model_v3.h5')

# Define emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to process the image and make predictions
def predict_emotion(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to match model's input size
    resized_img = cv2.resize(gray_image, (48, 48))
    # Reshape to add batch dimension
    reshaped_img = np.reshape(resized_img, (1, 48, 48, 1))
    # Normalize pixel values
    reshaped_img = reshaped_img / 255.0
    
    # Predict the emotion
    result = pretrained_model.predict(reshaped_img)
    label = np.argmax(result, axis=1)[0]
    
    return emotion_labels[label]

# Streamlit application
st.title("Emotion Recognition App")
st.write("This application uses a pre-trained deep learning model to recognize emotions either from your webcam feed or uploaded images.")

# Sidebar for navigation
option = st.sidebar.selectbox('Choose an option:', ('Real-Time Webcam', 'Upload Image'))

if option == 'Real-Time Webcam':
    st.write("Real-Time Emotion Recognition from Webcam Feed")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    if run:
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture frame from the webcam. Please ensure your webcam is connected.")
                break
            
            # Detect face
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)

            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Predict emotion
                face = frame[y:y+h, x:x+w]
                emotion = predict_emotion(face)
                
                # Annotate the image with the emotion label
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Display the image with annotations in Streamlit
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Stop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()

elif option == 'Upload Image':
    st.write("Upload an Image for Emotion Recognition")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display the uploaded image
        st.image(img_array, caption='Uploaded Image', use_column_width=True)
        
        # Predict emotion
        emotion = predict_emotion(img_array)
        
        # Display the prediction
        st.write(f"Predicted Emotion: **{emotion}**")
