import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import cv2

# Step 1: Load and inspect the data
georges_folder = r'george_test_task\george'  # Update with the path to the georges folder
no_georges_folder = r'george_test_task\no_george'  # Update with the path to the no_georges folder

georges_images = [os.path.join(georges_folder, img) for img in os.listdir(georges_folder) if img.endswith('.jpg')]
no_georges_images = [os.path.join(no_georges_folder, img) for img in os.listdir(no_georges_folder) if img.endswith('.jpg')]

georges_df = pd.DataFrame({'image_path': georges_images, 'target': 1})
no_georges_df = pd.DataFrame({'image_path': no_georges_images, 'target': 0})

combined_df = pd.concat([georges_df, no_georges_df], ignore_index=True)

# Step 2: Prepare training and testing data
X_paths = combined_df['image_path']
y = combined_df['target']

# Load and resize images using OpenCV
image_size = (100, 100)  # Define the desired image size
X_images = []
for img_path in X_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, image_size)  # Resize the image
    X_images.append(img.flatten())  # Flatten the image and add to X_images

# Convert list of flattened images to numpy array
X = np.array(X_images)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Choose a model (Random Forest Classifier)
clf = RandomForestClassifier(random_state=42)

# Step 4: Train the model
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Step 6: Save the model
joblib.dump(clf, 'george_classifier.pkl')

# Example usage:
def classify_image(image_path, model_path='george_classifier.pkl'):
    clf = joblib.load(model_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, image_size)  # Resize the image
    img_flat = img.flatten()  # Flatten the image
    # Preprocess the image if needed
    prediction = clf.predict([img_flat])
    return "St. George detected" if prediction[0] == 1 else "No St. George detected"

# Example usage:
image_path = r'george_test_task\george\0a5f7b5996063605dd05887ef4d31855.jpg'  # Update with the path to the test image
result = classify_image(image_path)
print(result)
