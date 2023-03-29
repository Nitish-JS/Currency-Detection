import cv2
import streamlit as st
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


model = load_model('currency_Classifier.h5')
class_names = ['1Hundrednote', '2Hundrednote', '2Thousandnote',
               '5Hundrednote', 'Fiftynote', 'Tennote', 'Twentynote']


def detect_currency_note(img):
    img = cv2.resize(img, (180, 180))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)
    result = np.argmax(model.predict(test_image), axis=1)
    prediction = class_names[result[0]]
    return prediction

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty("rate", 125)
    engine.runAndWait()


def app():
    st.title("Currency Note Detection")
    st.write("Upload an image of a currency note to detect its denomination")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        # img = load_img(uploaded_file, target_size=(180, 180))
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Detect the currency note
        prediction = detect_currency_note(img)

        # Display the result
        st.write("It is ₹"+prediction)
        if(prediction == "5Hundrednote"):
            speak("It is ₹ five hundred note")
        elif(prediction == "2Hundrednote"):
            speak("It is ₹ two hundred note")
        elif(prediction=="1Hundrednote"):
            speak("It is ₹ one hundred note")
        elif(prediction == "2Thousandnote"):
            speak("It is ₹ two thousand note")
        elif(prediction == "Fiftynote"):
            speak("It is ₹ fifty note")
        elif(prediction == "Tennote"):
            speak("It is ₹ Ten note")
        elif(prediction == "Twentynote"):
            speak("It is ₹ twenty note")


# Run the app
if __name__ == '__main__':
    app()
