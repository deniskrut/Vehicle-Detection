import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from extract_features import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# Read in cars and notcars
cars = glob.glob('vehicles_smallset/cars[0-9]/*.jpeg')
notcars = glob.glob('non-vehicles_smallset/notcars[0-9]/*.jpeg')

car_features = extract_features_standard(cars)
notcar_features = extract_features_standard(notcars)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

joblib.dump(svc, 'car_classifier.pkl')
joblib.dump(X_scaler, 'feature_scaler.pkl')

