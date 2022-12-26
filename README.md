# MI
Performance of classifiers while classifying Motor Imagery
# System imports
import re
import warnings
# Data science imports
import pandas as pd 
import numpy as np 
# Visualization imports
import matplotlib.pyplot as plt 
# sklearn imports
import sklearn.model_selection 
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
import sklearn.discriminant_analysis
import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
!pip install tensorflow 
import keras
# Helper functions to compute F1-Score, accuracy for TF code
# Taken from: https://medium.com/@aakashgoel12/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
# and https://www.kaggle.com/code/anshumand0/anshuman-dewangan-ucsd-neural-data-challenge
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return precision

def recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return recall

def acc(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    true_negatives=len(y_pred) - ((possible_positives + predicted_positives)-true_positives)
    accuracy = (true_positives + true_negatives)/len(y_pred)
    return accuracy            

# Helper class to train sklearn gridsearchcv models & report metrics
class gridsearchcv_model:
#   model: saved model
#   name: name for model
#   train, val: object with {name, predictions, mse OR accuracy} 

    def __init__(self, model, X_train, Y_train, X_val, Y_val, parameter_matrix={}, is_classification=False, cv=4):
        self.is_classification = is_classification
        self.train_model(model, X_train, Y_train, X_val, Y_val, parameter_matrix, cv)
        
    # Trains model using a training set and predicts a validation set
    def train_model(self, model, X_train, Y_train, X_val, Y_val, parameter_matrix={}, cv=4):
        if self.is_classification:
            ml_model = sklearn.model_selection.GridSearchCV(model, parameter_matrix, cv=cv, scoring='f1')
        else:
             ml_model = sklearn.model_selection.GridSearchCV(model, parameter_matrix, cv=cv, scoring='neg_mean_squared_error')
        
        ml_model.fit(X_train, Y_train)
        
        self.model = ml_model.best_estimator_
        self.name = re.compile("(.*?)\s*\(").match(str(self.model)).group(1)
        
        self.train = {'name': 'train'}
        self.val = {'name': 'val'}
        
        self.calculate_error(self.train, X_train, Y_train, self.train['name'])
        self.calculate_error(self.val, X_val, Y_val, self.val['name'])
        
        return ml_model
    
    def calculate_error(self, var, X_set, Y_set, name):
        var['name'] = name
        var['predictions'] = self.model.predict(X_set)
        
        if self.is_classification:
            var['accuracy'] = sklearn.metrics.f1_score(Y_set, var['predictions'])
        else:
            var['mse'] = sklearn.metrics.mean_squared_error(Y_set, var['predictions'])
        self.print_error(var)
        
    # Prints error metrics
    def print_error(self, var):
        print(self.name + ' ('+ var['name'] + ')')
        
        if self.is_classification:
            print("Accuracy: %0.4f" % var['accuracy'])
        else:
            print("MSE: %0.4f" % var['mse'])

    # Load train and test data into dataframes
   data_path = "C:/Users/shubh/Downloads/ucsd-neural-data-challenge/data"
   df_train_original = pd.read_pickle ("C:/Users/shubh/Downloads/ucsd-neural-data-challenge/data/epoched_train.pkl")
   df_test = pd.read_pickle ("C:/Users/shubh/Downloads/ucsd-neural-data-challenge/data/epoched_test.pkl")
   #df_train_original = pd.read_pickle(data_path + "epoched_train.pkl")
   #df_test = pd.read_pickle(data_path + "epoched_test.pkl")        

   # Create column 'pid' which is the patient ID 1 through 9
df_train_original['pid'] = [int(df_train_original['patient_id'][x][2]) for x in range(len(df_train_original))]
df_test['pid'] = [int(df_test['patient_id'][x][2]) for x in range(len(df_test))]

# Create column 'trial_id' which is the trial 1 through 3
df_train_original['trial_id'] = [int(df_train_original['patient_id'][x][-2]) for x in range(len(df_train_original))]
df_test['trial_id'] = [int(df_test['patient_id'][x][-2]) for x in range(len(df_test))]


# Use trials 1&2 for training, trial 3 for validation (mirrors process to create Kaggle test set)
df_train = df_train_original[df_train_original['trial_id']!=3]
df_train = df_train.reindex(np.random.permutation(df_train.index)).reset_index(drop=True)

df_val = df_train_original[df_train_original['trial_id']==3]
df_val = df_val.reindex(np.random.permutation(df_val.index)).reset_index(drop=True)

# Augment training data by adding Gaussian noise
for _ in range(1): # Optional for loop to repeat this process multiple times; however, running loop once resulted in most optimal accuracy
    df_train_augment = df_train.copy()
    for x in ['C3', 'Cz', 'C4', 'EOG:ch01', 'EOG:ch02','EOG:ch03']:
        df_train_augment[x] += np.random.normal(0,1)

    df_train = pd.concat([df_train, df_train_augment])

# Prepare data for training across all subjects
y_train = df_train["event_type"].values.astype(float)
y_val = df_val["event_type"].values.astype(float)

X_train = df_train.drop(["patient_id", "start_time", "event_type", "pid", "trial_id"], axis=1)
X_val = df_val.drop(["patient_id", "start_time", "event_type", "pid", "trial_id"], axis=1)
X_test = df_test.drop(["patient_id", "start_time", "pid", "trial_id"], axis=1)

# Concatenate the data for sklearn & neural net models
x_train_nn = np.array(X_train.apply(lambda x:np.concatenate(x), axis=1).values.tolist())
x_val_nn = np.array(X_val.apply(lambda x:np.concatenate(x), axis=1).values.tolist())
x_test_nn = np.array(X_test.apply(lambda x:np.concatenate(x), axis=1).values.tolist())


# Regular neural net
neural_network = keras.Sequential([Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0001)),
                                   Dropout(0.2),
                                   Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0001)),
                                   Dropout(0.2),
                                   Dense(1, activation="sigmoid")])

neural_network.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy',get_f1,precision,recall])

history = neural_network.fit(x_train_nn, y_train, epochs=10, batch_size=64)


print(neural_network.summary())

# Evaluate on validation set
neural_network.evaluate(x_val_nn, y_val)


# Stack the data for CNN models
x_train_cnn = np.array(X_train.apply(lambda x:np.stack(x, axis=-1), axis=1).values.tolist())
x_train_cnn = x_train_cnn.reshape(list(x_train_cnn.shape)+[1])

x_val_cnn = np.array(X_val.apply(lambda x:np.stack(x, axis=-1), axis=1).values.tolist())
x_val_cnn = x_val_cnn.reshape(list(x_val_cnn.shape)+[1])

cnn = keras.Sequential([Conv2D(32, (3, 3), activation="relu", input_shape=(1000, 6, 1)),
                        Conv2D(64, (3, 3), activation="relu"),
                        Flatten(),
                        #Dense(64, activation="relu"),
                        #Dropout(0.2),
                        #Dense(32, activation="relu"),
                        #Dense(1, activation="sigmoid")])
                        Dense(256, activation="relu"),
                        Dropout(0.2),
                        Dense(128, activation="relu"),
                        Dense(1, activation="sigmoid")])

cnn.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy',get_f1,precision,recall,acc])

history = cnn.fit(x_train_cnn, y_train, epochs=10, batch_size=64)

print(cnn.summary())

# Evaluate on validation set
cnn.evaluate(x_val_cnn, y_val)

# Reshape data for transfer learning models
x_train_transfer = x_train_cnn.reshape((len(x_train_cnn), 50, 40, 3))
x_val_transfer = x_val_cnn.reshape((len(x_val_cnn), 50, 40, 3))


resnet_weights_path = 'C:/Users/shubh/Downloads/archive/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

transfer_network = keras.Sequential([ResNet50(include_top=False, weights=resnet_weights_path, input_shape=(50,40,3)),
                                     Flatten(),
                                     Dense(1, activation="sigmoid")])

opt = Adam(lr=0.001)
transfer_network.layers[0].trainable = False

transfer_network.compile(optimizer=opt, loss='binary_crossentropy', metrics=[get_f1,precision,recall,'accuracy'])

history = transfer_network.fit(x_train_transfer, y_train, epochs=20, batch_size=64)


# Evaluate on validation set
transfer_network.evaluate(x_val_transfer, y_val)


# Reshape data for autoencoder
latent_factors = 25

x_train_ae = x_train_nn.reshape(-1,latent_factors)
x_val_ae = x_val_nn.reshape(-1,latent_factors)


# Autoencoder
autoencoder = keras.Sequential([Dense(latent_factors, activation="relu")])

autoencoder.compile(optimizer=Adam(lr=0.001), loss=keras.losses.MeanSquaredError())

history = autoencoder.fit(x_train_ae, x_train_ae, epochs=10, batch_size=64)