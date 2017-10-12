
# coding: utf-8

# ### Read the log file and partition the data into training and validation set

# In[1]:

import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import sklearn

# Reading the driving log into an array
samples  = []
with open("./training_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
# Split samples between training and validation        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Number of training samples: {t}".format(t = len(train_samples)) )
print("Number of validation samples: {v}".format(v=len(validation_samples)) )


# ### Helper functions

# In[2]:

'''
For each entry in the log (batch), picture of the center, left, and right side is appended to images[]
along with the stearing angle entry into measurements[].
'''
def getImages(batch):
    location = "./training_data/IMG/"
    
    images, measurements = [],[]
    for batch_sample in batch:
        for j in range(3):
            filename = batch_sample[j].split('/')[-1]   # j (1-3) is for central, left and right side images 
            current_path = location + filename
            image = cv2.imread(current_path)
            images.append(image)
            if(j==0):
                measurements.append(float(batch_sample[3]))
            if(j==1):
                measurements.append(float(batch_sample[3]) + 0.2)  # add
            if(j==2):
                measurements.append(float(batch_sample[3]) - 0.2)  # subtract 
    return images, measurements


'''
Augmenting mirror images to account for right side turn and increase the training data
'''
def augment_images(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * (-1.0))
    return augmented_images, augmented_measurements


# In[3]:

'''
The generator function is used to feed the samples of batch size into training model.
This makes the process of feeding the input a lot faster.
'''

def generator(samples, batch_size):
    num_samples = len(samples)
    while(1):
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # Define a batch
            batch = samples[offset: (offset+batch_size)]
            
            # get Images from new path
            images, measurements = getImages(batch)
                 
            # augment images to accomodate training for 'right turns'
            images, measurements = augment_images(images, measurements)
            
            X = np.array(images)
            y = np.array(measurements)
            # Shuffle the data
            yield sklearn.utils.shuffle(X, y) 
        
 


# ### CNN model

# In[4]:

# NVidia self driving car Model


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

batch_size = 128
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer = 'adam')

model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*6,
                    validation_data=validation_generator, nb_val_samples=len(validation_samples)*6,
                    nb_epoch=5, verbose=1)


model.save("model.h5")

