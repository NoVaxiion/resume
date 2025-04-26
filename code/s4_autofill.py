"""
Autofill Chrome form fields based on OCR-detected field labels, using trained model predictions.

JUST KNOW MY (KENNETH's) INFORMATION IS HARDCODED AT THE MOMENT
YOU CAN IGNORE THE WARNING AAS WE'RE ONLY PREDICTING THE TEXT
"""

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tensorflow.keras.layers import TextVectorization
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
import tensorflow as tf
import numpy as np
import pickle
import time

#Load the trained model
model = tf.keras.models.load_model('../models/field_classifier_model.h5')

#Load label encoder
with open('../models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

#Load vectorizer vocabulary and rebuild vectorizer
with open('../models/vectorizer_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

vectorizer = TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=20)
vectorizer.set_vocabulary(vocab)

#Personal info for autofill
personal_info = {
    "first_name": "Kenneth",
    "last_name": "Maeda",
    "full_name": "Kenneth Maeda",
    "email": "kenneth.maeda012403@gmail.com",
    "phone_number": "2035703208",
    "city": "Stamford",
    "state": "CT",
    "zipcode": "06902",
    "country": "United States",
    "linkedin": "https://linkedin.com/in/kenneth-maeda"
}

#Predict function
def predict_field(field_text):
    field_tensor = tf.convert_to_tensor([field_text])
    field_vectorized = vectorizer(field_tensor)
    preds = model.predict(field_vectorized)
    pred_class = np.argmax(preds, axis=1)
    pred_label = label_encoder.inverse_transform(pred_class)[0]
    return pred_label

#Ask user for URL
url = input("Enter the URL to autofill: ").strip()

#Launch Chrome
options = Options()
options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=options)
driver.get(url)

wait = WebDriverWait(driver, 15)

#Wait for form to load
wait.until(EC.presence_of_element_located((By.TAG_NAME, "input")))

#Find input fields
inputs = driver.find_elements(By.TAG_NAME, "input")

#Try to fill text inputs
for input_element in inputs:
    try:
        label_element = input_element.find_element(By.XPATH, "./ancestor::div[contains(@class, 'field')]//label")
        field_label = label_element.text.strip()

        if field_label:
            predicted_label = predict_field(field_label)
            print(f"Field: {field_label} ➔ Predicted: {predicted_label}")

            #If predicted label matches our personal_info keys, fill it
            if predicted_label in personal_info:
                value_to_fill = personal_info[predicted_label]
                input_element.send_keys(value_to_fill)
                time.sleep(0.5)  #Small delay so browser doesn't crash

    except Exception as e:
        #Skip fields without label or problematic ones
        continue

print("Finished autofilling known fields!")

