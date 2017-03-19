import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load dataset
df = read_csv('housing.csv',delim_whitespace=True, header=None)
dt = df.values

X = dt[:,0:13]
y = dt[:,13]

# Define the baseline model
def baseline_model():
    
    model = Sequential()
    model.add( Dense(13, input_dim=13, init='normal', activation='relu') )
    model.add( Dense(1, init='normal') )

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Set seed 
seed = 7
np.random.seed(seed)

# define estimator
estimators = []
estimators.append( ('standardize', StandardScaler()) )
estimators.append( ('regressor', KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)) )
pipeline = Pipeline(estimators)

# In evaluation, do 10 fold x-val
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print ("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


