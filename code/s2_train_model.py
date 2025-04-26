"""
DATA MUST BE CLEANED FIRST
Train the field label prediction model using Keras TextVectorization and save the model, label encoder, and vectorizer vocabulary.
"""

from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
import pandas as pd
import pickle

#Load cleaned dataset
data = pd.read_csv('../data/ocr_labels.csv')


#Prepare the data
data = data[data['target_label'].notnull() & (data['target_label'].str.strip() != '')]
X = data['field_text'].astype(str)
y = data['target_label'].astype(str)

#Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Save the label encoder
with open('../models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

#Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#Create and adapt the vectorizer
vectorizer = TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=20)
vectorizer.adapt(X_train)

#Save the vectorizer vocabulary
vocab = vectorizer.get_vocabulary()
with open('../models/vectorizer_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

#Vectorize training data
X_train_vectorized = vectorizer(X_train)
X_val_vectorized = vectorizer(X_val)

#Build the model (without embedding the vectorizer inside)
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, mask_zero=True),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#Train the model
callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
history = model.fit(
    X_train_vectorized,
    y_train,
    validation_data=(X_val_vectorized, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

#Save the model
model.save('../models/field_classifier_model.h5')

print("Model, Label Encoder, and Vectorizer vocabulary saved successfully.")
