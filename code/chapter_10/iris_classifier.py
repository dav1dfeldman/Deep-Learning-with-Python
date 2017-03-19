
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# Setting up seed to be able to reproduce
seed = 7
np.random.seed(seed)

# Load and prep data
df = pd.read_csv("iris.csv", header = None)
dt = df.values # convert to numpy representation of the data
X = dt[:,0:4].astype(float)
y = dt[:,4]

# Next we want to take our target (the flower type) and encode
# it such that the encoding is reshaped into N-type columns with
# a boolean.  It's mentioned this is better for NN problems to 
# handle
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)               # This assigns it to numerical value
dummy_y   = np_utils.to_categorical(encoded_Y) # This reshapes to matrix

# Define the baseline model
def base_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=base_model, nb_epoch=200, batch_size=5, verbose=0)

# Now use k-fold CV to evaluate
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print ("Accuracy %.2f%% (%.2f%%)" %(results.mean()*100, results.std()*100))

