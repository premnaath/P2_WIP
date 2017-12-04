# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


# TODO: Fill this in based on where you saved the training and testing data

training_file = '/home/premnaath/stuff/udacity/P/P2/traffic-signs-data/train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

print(y_train.shape)

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.max(y_train) + 1 # Zeroo based, so +1

print("Number of training examples =", n_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Visualization
full_range = np.arange(0, n_train, 1)
rand_value = random.choice(full_range)

img_copy = np.copy(X_train[rand_value])
grey = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

plt.imshow(grey, cmap='gray')
plt.show()
