#!/usr/bin/env python3
"""
Universal Autofill Script
Automatically fills and interacts with form fields on any job application page.
"""

import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# —— Load ML model & components —— #
model = tf.keras.models.load_model('../models/field_classifier_model.h5')
with open('../models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('../models/vectorizer_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vectorizer = TextVectorization(
    max_tokens=5000,
    output_mode='int',
    output_sequence_length=20
)
vectorizer.set_vocabulary(vocab)

# —— Your hard-coded personal info —— #
personal_info = {
    "first_name":      "Kenneth",
    "last_name":       "Maeda",
    "full_name":       "Kenneth Maeda",
    "email":           "kenneth.maeda012403@gmail.com",
    "phone_number":    "2035703208",
    "address":         "47 Stephen St",
    "city":            "Stamford",
    "state":           "CT",
    "zipcode":         "06902",
    "country":         "United States",
    "linkedin":        "https://linkedin.com/in/kenneth-maeda",
    "portfolio_link":  "https://kenneth-maeda.dev",
    "linkedin_profile":"https://linkedin.com/in/kenneth-maeda",
    "resume_upload":   "/absolute/path/to/your/resume.pdf"
}

# —— Heuristic override before ML —— #
def heuristic_label(elem):
    typ = (elem.get_attribute('type') or '').lower()
    nm  = (elem.get_attribute('name') or '').lower()
    idx = (elem.get_attribute('id')   or '').lower()
    ph  = (elem.get_attribute('placeholder') or '').lower()

    if typ == 'email':    return 'email'
    if typ == 'tel':      return 'phone_number'
    if typ == 'url':      return 'portfolio_link'
    if typ == 'file':     return 'resume_upload'

    for key in ('first_name','last_name','email','phone_number',
                'address','city','state','zipcode'):
        token = key.replace('_','')
        if token in nm or token in idx:
            return key

    if 'start typing' in ph or 'city' in ph: return 'city'
    if '@' in ph:                            return 'email'
    if any(c.isdigit() for c in ph) and '-' in ph:
        return 'phone_number'

    return None

# —— ML predictor fallback —— #
def predict_field(label_text):
    tensor = tf.convert_to_tensor([label_text])
    vect   = vectorizer(tensor)
    preds  = model.predict(vect)
    idx    = np.argmax(preds, axis=1)
    return label_encoder.inverse_transform(idx)[0]

# —— Robust click + type helper —— #
def click_and_type(elem, text, driver):
    try:
        elem.click()
    except:
        try:
            driver.execute_script('arguments[0].click();', elem)
        except:
            ActionChains(driver).move_to_element(elem).click().perform()
    time.sleep(0.5)
    elem.clear()
    elem.send_keys(text)
    time.sleep(0.5)

# —— Main Autofill Routine —— #
def main():
    url = input("Enter the URL to autofill: ").strip()
    options = Options()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'input')))

    inputs = driver.find_elements(By.TAG_NAME, 'input')
    for elem in inputs:
        try:
            # 1) Skip non-interactable
            if not (elem.is_displayed() and elem.is_enabled()):
                continue

            # 2) Only these input types
            typ = (elem.get_attribute('type') or '').lower()
            if typ not in {'text','email','tel','url','number','search','password','file'}:
                continue

            # —— START LABEL DETECTION —— #
            label_text = ''

            # a) <label> wrapping the input?
            try:
                lbl = elem.find_element(By.XPATH, 'ancestor::label')
                label_text = lbl.text.strip()
            except:
                pass

            # b) <label for="id"> elsewhere?
            if not label_text:
                iid = elem.get_attribute('id')
                if iid:
                    try:
                        lbl = driver.find_element(By.CSS_SELECTOR, f"label[for='{iid}']")
                        label_text = lbl.text.strip()
                    except:
                        pass

            # c) Immediately preceding <label> sibling?
            if not label_text:
                try:
                    lbl = elem.find_element(By.XPATH, 'preceding-sibling::label[1]')
                    label_text = lbl.text.strip()
                except:
                    pass

            # d) Label just above in parent container?
            if not label_text:
                try:
                    lbl = elem.find_element(By.XPATH, 'parent::div/preceding-sibling::label[1]')
                    label_text = lbl.text.strip()
                except:
                    pass

            # e) Fallback to placeholder / aria-label
            if not label_text:
                label_text = (
                    elem.get_attribute('placeholder')
                    or elem.get_attribute('aria-label')
                    or ''
                )

            # If still no label, skip
            if not label_text:
                continue
            # —— END LABEL DETECTION —— #

            # 3) Decide key via heuristic or ML
            key = heuristic_label(elem) or predict_field(label_text)
            print(f"Field: '{label_text}' → Key: {key}")
            if key not in personal_info:
                continue
            val = personal_info[key]

            # 4) Fill based on type/key
            if typ == 'file':
                elem.send_keys(val)            # Resume upload
            elif key == 'city':
                click_and_type(elem, val, driver)
                elem.send_keys(Keys.ARROW_DOWN)
                elem.send_keys(Keys.ENTER)
            else:
                click_and_type(elem, val, driver)

        except Exception as e:
            print(f"⚠️ Skipping field due to error: {e}")
            continue

    print("✅ Autofill complete.")

if __name__ == '__main__':
    main()



# """
# Autofill Chrome form fields based on OCR-detected field labels, using trained model predictions.

# JUST KNOW MY (KENNETH's) INFORMATION IS HARDCODED AT THE MOMENT
# YOU CAN IGNORE THE WARNING AAS WE'RE ONLY PREDICTING THE TEXT
# """

# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import WebDriverWait
# from tensorflow.keras.layers import TextVectorization
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium import webdriver
# import tensorflow as tf
# import numpy as np
# import pickle
# import time

# #Load the trained model
# model = tf.keras.models.load_model('../models/field_classifier_model.h5')

# #Load label encoder
# with open('../models/label_encoder.pkl', 'rb') as f:
#     label_encoder = pickle.load(f)

# #Load vectorizer vocabulary and rebuild vectorizer
# with open('../models/vectorizer_vocab.pkl', 'rb') as f:
#     vocab = pickle.load(f)

# vectorizer = TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=20)
# vectorizer.set_vocabulary(vocab)

# #Personal info for autofill
# personal_info = {
#     "first_name": "Kenneth",
#     "last_name": "Maeda",
#     "full_name": "Kenneth Maeda",
#     "email": "kenneth.maeda012403@gmail.com",
#     "phone_number": "2035703208",
#     "city": "Stamford",
#     "state": "CT",
#     "zipcode": "06902",
#     "country": "United States",
#     "linkedin": "https://linkedin.com/in/kenneth-maeda"
# }

# #Predict function
# def predict_field(field_text):
#     field_tensor = tf.convert_to_tensor([field_text])
#     field_vectorized = vectorizer(field_tensor)
#     preds = model.predict(field_vectorized)
#     pred_class = np.argmax(preds, axis=1)
#     pred_label = label_encoder.inverse_transform(pred_class)[0]
#     return pred_label

# #Ask user for URL
# url = input("Enter the URL to autofill: ").strip()

# #Launch Chrome
# options = Options()
# options.add_experimental_option("detach", True)

# driver = webdriver.Chrome(options=options)
# driver.get(url)

# wait = WebDriverWait(driver, 15)

# #Wait for form to load
# wait.until(EC.presence_of_element_located((By.TAG_NAME, "input")))

# #Find input fields
# inputs = driver.find_elements(By.TAG_NAME, "input")

# #Try to fill text inputs
# for input_element in inputs:
#     try:
#         label_element = input_element.find_element(By.XPATH, "./ancestor::div[contains(@class, 'field')]//label")
#         field_label = label_element.text.strip()

#         if field_label:
#             predicted_label = predict_field(field_label)
#             print(f"Field: {field_label} ➔ Predicted: {predicted_label}")

#             #If predicted label matches our personal_info keys, fill it
#             if predicted_label in personal_info:
#                 value_to_fill = personal_info[predicted_label]
#                 input_element.send_keys(value_to_fill)
#                 time.sleep(0.5)  #Small delay so browser doesn't crash

#     except Exception as e:
#         #Skip fields without label or problematic ones
#         continue

# print("Finished autofilling known fields!")

