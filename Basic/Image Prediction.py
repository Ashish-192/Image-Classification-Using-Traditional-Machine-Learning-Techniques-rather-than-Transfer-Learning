import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm

# Load digits dataset
digits = datasets.load_digits()

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
fig.suptitle('Sample Images from the Digits Dataset')

for i, ax in enumerate(axes.flat):
  ax.imshow(digits.images[i], cmap='gray')
  ax.set_title(f'Digit: {digits.target[i]}')
  ax.axis('off')

plt.show()

#Image Flatten
data = digits.images.reshape((len(digits.images), -1))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, random_state=42)

# Standardize
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM classifier
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_train, y_train)
# Code snippet for model evaluation
from sklearn import metrics

# predictions
y_pred = clf.predict(X_test)

# Evaluate
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')
# Code result visualization
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
fig.suptitle(f'Model Predictions (Accuracy: {accuracy:.2%})')

for i, ax in enumerate(axes.flat):
  ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
  ax.set_title(f'Prediction: {y_pred[i]}')
  ax.axis('off')

plt.show()
