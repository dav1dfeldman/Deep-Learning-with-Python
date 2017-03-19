
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


# Create model function
def create_model(optimizer='rmsprop',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, init=init, activation='relu'))
    model.add(Dense(8, init=init, activation='relu'))
    model.add(Dense(1, init=init, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Set seed and get going
seed = 7
np.random.seed(seed)

dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=",")
X = dataset[:,0:8]
y = dataset[:,8]

# Create model with wrapper
model = KerasClassifier(build_fn=create_model, verbose=0)

# Setup what to gridsearch for
optimizer = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50,100,150]
batches = [5,10,20]

# Run grid search
param_grid = dict(optimizer=optimizer, nb_epoch=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X,y)

# Print all the results
print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds  = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print ("%f (%f) with: %r" % (mean, stdev, param))
