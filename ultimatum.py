import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
#<---- Preprocessing ---->

def extract_features(img):
    # Resize the image to a fixed size
    img = cv2.resize(img, (100, 100))

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a mask to extract the tomato color
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2

    # Extract the mean color of the tomato
    mean_color = cv2.mean(img, mask=mask)[0:3]

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    c = max(contours, key=cv2.contourArea)

    # Extract the shape features of the tomato
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / perimeter ** 2

    return mean_color, circularity

# Define the input and output directories
input_dir = 'C:/Users/sriir/study material/ai project/Riped and Unriped tomato Dataset/Images'
output_dir = 'C:/Users/sriir/study material/ai project/result'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the input directory and extract the features of each image
for filename in os.listdir(input_dir):
    # Load the image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # Extract the color and shape features of the tomato
    color_features, shape_features = extract_features(img)

    # Save the features to a text file
    output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
    with open(output_path,'w') as f:
        f.write(f'{color_features[0]},{color_features[1]},{color_features[2]},{shape_features}\n')


#<-----Training model on the preprocessed data----->

# Define the input and output directories
input_dir = 'C:/Users/sriir/study material/ai project/result'
output_dir = 'C:/Users/sriir/study material/ai project'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the features and labels from the output files
features = []
labels = []
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            values = line.strip().split(',')
            mean_color = np.array([float(values[0]), float(values[1]), float(values[2])])
            circularity = float(values[3])
            features.append(np.concatenate((mean_color, [circularity])))
            if 'unriped' in filename:
                labels.append(0)
               # print("labeled as unripe")
               
            elif('riped'):
                labels.append(1)
               # print("labeled as ripe")
            line = f.readline()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save the trained model to a file
model_path = os.path.join(output_dir, 'random_forest.joblib')
joblib.dump(clf, model_path)
print(f'Trained model saved to {model_path}')
