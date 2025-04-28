"""
THE TARGET_LABELS MUST BE MANUALLY FILLED OUT BEFORE RUNNING THIS TO SEE THE PREDICTIONS MADE
Batch predict clean field labels from a list of OCR fields.
"""
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

#Load the trained model
model = tf.keras.models.load_model('../models/field_classifier_model.h5')

#Load the label encoder
with open('../models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

#Load the saved vocabulary and rebuild vectorizer
with open('../models/vectorizer_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

vectorizer = TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=20)
vectorizer.set_vocabulary(vocab)

#Function to predict a list of OCR fields
def batch_predict(fields_list):
    fields_tensor = tf.convert_to_tensor(fields_list)
    fields_vectorized = vectorizer(fields_tensor)
    preds = model.predict(fields_vectorized)
    pred_classes = np.argmax(preds, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_classes)
    return pred_labels

if __name__ == "__main__":
    print("Model loaded. Ready for batch field predictions.")

    #Example OCR field list to test
    fields_to_predict = [
        "First Name",
        "Last Name",
        "Email Address",
        "Phone Number",
        "Upload Resume",
        "LinkedIn Profile",
        "City",
        "State",
        "Postal Code",
        "Country",
        "Submit Application",
    ]

    predictions = batch_predict(fields_to_predict)

    #Display results nicely
    results = pd.DataFrame({
        "Field Text": fields_to_predict,
        "Predicted Label": predictions
    })

    print(results.to_string(index=False))
