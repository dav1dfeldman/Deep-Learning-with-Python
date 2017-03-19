from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

# Define the model
def create_model():
    
    model = Sequential()
    model.add( Dense(12, input_dim=8, init='uniform', activation='relu') )
    model.add( Dense(8, init='uniform', activation='relu') )
    model.add( Dense(1, init='uniform', activation='sigmoid') )

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Set seed and load data
seed = 7
np.random.seed(seed)

dt = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dt[:,0:8]
y = dt[:,8]

# Create classifier
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=0)

# Evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)

print (results.mean())
