# Install necessary packages
#!pip install --upgrade gspread pandas tensorflow openpyxl
#!pip install --upgrade gspread google-auth


import gspread
from google.colab import auth
from google.auth.transport.requests import Request
from google.auth import default

# Authenticate
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

# Use your Google Sheet URL here
sheet_url = 'https://docs.google.com/spreadheets/d/1F4FSBxEpiTnAEffljl1e91tJxKRmEY/edit#gid=0'

# Open the spreadsheet
spreadsheet = gc.open_by_url(sheet_url)

# Access the first sheet
worksheet = spreadsheet.get_worksheet(0)

# Read all rows into a DataFrame
import pandas as pd
clinical_data = pd.DataFrame(worksheet.get_all_records())

# Show result
print(clinical_data.head())


# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Step 2: Load and preprocess clinical data
sheet_csv_url = 'https://docs.google.com/spreadsheets/d/1F4FSBxEpiTnAEffljl1oXjc6jzIvzeJe91tJxKRm4EY/export?format=csv'
clinical_data = pd.read_csv(sheet_csv_url)

clinical_data.columns = clinical_data.columns.str.strip()
clinical_data.replace('MISSING', np.nan, inplace=True)
clinical_data['Weight_kg'] = pd.to_numeric(clinical_data['Weight_kg'], errors='coerce')
clinical_data['GaitSpeed(m/sec)'] = pd.to_numeric(clinical_data['GaitSpeed(m/sec)'], errors='coerce')
clinical_data.fillna(clinical_data.median(numeric_only=True), inplace=True)

clinical_data['gender'] = clinical_data['gender'].map({'f': 0, 'm': 1})
label_encoder = LabelEncoder()
clinical_data['DiseasesTypEncoded'] = label_encoder.fit_transform(clinical_data['DiseasesTyp'])
class_names = label_encoder.classes_

# Step 3: Balance classes (full dataset)
majority_class = clinical_data[clinical_data['DiseasesTypEncoded'] == clinical_data['DiseasesTypEncoded'].value_counts().idxmax()]
minority_class = clinical_data[clinical_data['DiseasesTypEncoded'] != clinical_data['DiseasesTypEncoded'].value_counts().idxmax()]
oversampled_minority = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
balanced_clinical_data = pd.concat([majority_class, oversampled_minority])

# Step 4: Prepare input data
X_clinical = balanced_clinical_data[['AGE(YRS)', 'HEIGHT(meters)', 'Weight_kg', 'gender', 'GaitSpeed(m/sec)', 'Duration/Severity']].values
y = balanced_clinical_data['DiseasesTypEncoded'].values

scaler = StandardScaler()
X_clinical = scaler.fit_transform(X_clinical)

X_clinical_train, X_clinical_test, y_train, y_test = train_test_split(X_clinical, y, test_size=0.2, random_state=42)

# Step 5: Load image data
image_data_path = '/content/drive/MyDrive/Google Colab/dataset'
image_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_image_gen = image_gen.flow_from_directory(image_data_path, target_size=(224, 224), batch_size=16, class_mode='categorical', subset='training', shuffle=True)
test_image_gen = image_gen.flow_from_directory(image_data_path, target_size=(224, 224), batch_size=16, class_mode='categorical', subset='validation', shuffle=False)
num_classes = len(train_image_gen.class_indices)

# Step 6: Combined generator
def combined_generator(image_gen, clinical_data, clinical_labels):
    while True:
        images, _ = next(image_gen)
        batch_size = images.shape[0]
        idx = image_gen.index_array[:batch_size] % len(clinical_data)
        clinical_batch = tf.convert_to_tensor(clinical_data[idx], dtype=tf.float32)
        clinical_label_batch = tf.convert_to_tensor(clinical_labels[idx], dtype=tf.int32)
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        yield (images, clinical_batch), tf.one_hot(clinical_label_batch, num_classes)

output_signature = (
    (tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(None, 6), dtype=tf.float32)),
    tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(lambda: combined_generator(train_image_gen, X_clinical_train, y_train), output_signature=output_signature)
test_dataset = tf.data.Dataset.from_generator(lambda: combined_generator(test_image_gen, X_clinical_test, y_test), output_signature=output_signature)

# Step 7: Model architecture
image_input = Input(shape=(224, 224, 3), name="image_input")
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
image_output = Dense(64, activation='relu')(x)

clinical_input = Input(shape=(6,), name="clinical_input")
y = Dense(64, activation='relu')(clinical_input)

combined = concatenate([image_output, y])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.5)(combined)
z = Dense(num_classes, activation='softmax')(combined)

model = Model(inputs=[image_input, clinical_input], outputs=z)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 8: Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_image_gen) // 5,
    validation_data=test_dataset,
    validation_steps=len(test_image_gen) // 2,
    epochs=20,
    callbacks=[early_stopping, lr_scheduler]
)

# Step 9: Evaluation
loss, accuracy = model.evaluate(test_dataset, steps=len(test_image_gen) // 5)
print(f"Test accuracy: {accuracy:.4f}")

# Step 10: Plot training metrics
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

# Step 11: Confusion Matrix
y_true = []
y_pred = []
steps = len(test_image_gen)

for i, ((image_batch, clinical_batch), label_batch) in enumerate(test_dataset.take(steps)):
    true = tf.argmax(label_batch, axis=1).numpy()
    preds = model.predict([image_batch, clinical_batch], verbose=0)
    pred = tf.argmax(preds, axis=1).numpy()
    y_true.extend(true)
    y_pred.extend(pred)

unique_classes = sorted(list(set(y_true) | set(y_pred)))
filtered_names = [class_names[i] for i in unique_classes]

cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=filtered_names, zero_division=0))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=filtered_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

