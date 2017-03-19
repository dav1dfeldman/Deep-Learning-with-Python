
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline # Why?

seed = 7
np.random.seed(seed)

# Load data
dt = pd.read_csv('sonar.csv',header=None).values
X = dt[:,0:60].astype(float)
y = dt[:,60]

# Encode data
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

# Setup model
def base_model():
    model = Sequential()
    model.add( Dense(60, input_dim=60, init='normal', activation='relu') )
    model.add( Dense(30, init='normal', activation='relu') )
    model.add( Dense(1, init='normal', activation='sigmoid') )
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Setup and evaluate
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=base_model, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold     = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results   = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print ("Baseline: %.2f%% (%.2f%%)" %(results.mean()*100, results.std()*100)) 
