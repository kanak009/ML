project_folder/
│p
├── templates/
│ └── upload.html
├── uploads.py # for temp uploaded images
├── model.py
├── server.py
├── requirements.txt
└── george_classifier.pkl # Trained model file
Above is Project Structure
LOCAL SETUP :
1. >> install -r requirements.txt
2. >> python server.py
Now, open a web browser and go to http://localhost:5000 to access your Flask web
application. You should see your interface, where you can upload images and get classification
results.
1. Data Loading and Preprocessing:
 - Loaded images from two folders: one containing images of St. George and the other
containing images of non-St. George.
 - Images were loaded using OpenCV and then converted to RGB format.
2. Data Preparation:
 - Created dataframes for St. George images and non-St. George images, adding a target
column to indicate the class.
 - Both dataframes were concatenated into a single dataframe for further processing.
3. Training and Testing Data Split:
 - The data was split into training and testing sets using `train_test_split` from
`sklearn.model_selection`.
4. Model Selection:
 - Chosen a Random Forest Classifier (`RandomForestClassifier`) as the model for this task.
5. Model Training:
 - The Random Forest Classifier was trained using the training data (`X_train`, `y_train`) using
the `fit` method.
6. Model Evaluation:
 - After training, the model was evaluated using the testing data (`X_test`, `y_test`).
 - Evaluation metrics such as accuracy, precision, recall, and F1-score were calculated using
`accuracy_score` and `classification_report` from `sklearn.metrics`.
7. Model Saving:
 - The trained model was saved using `joblib.dump` for future use (`george_classifier.pkl`).
8. Image Classification Function:
 - Defined a function `classify_image` that takes an image path as input and uses the trained
model to classify whether it contains St. George or not.
9. Flask Web Interface:
 - Created a Flask web application to serve as an interface for uploading images and getting
classification results.
 - The web application uses a POST request to send the uploaded image to the server for
classification.
 - The server then processes the image using the `classify_image` function and returns the
classification result to the client.
10. Interface Improvements:
 - Added a form on the web interface to allow users to upload images easily.
 - After classification, the result (St. George detected or not) is displayed on the web page
