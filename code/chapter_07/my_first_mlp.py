from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Fix randomseed for reproducibility
seed = 7
np.random.seed(seed)

# Load the data
dt = np.loadtxt("pima-indians-diabetes.csv",delimiter=',')

# Split into X and y
X = dt[:,0:8]
y = dt[:,8]

# Create the model
model = Sequential()
model.add( Dense(12, input_dim=8, init='uniform', activation='relu') )
model.add( Dense(8, init='uniform', activation='relu') )
model.add( Dense(1, init='uniform', activation='sigmoid') )

# Compile model
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# Fit
model.fit(X,y, nb_epoch=150, batch_size=10, validation_split=0.33)

# evaluate
scores = model.evaluate(X,y)
print ("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
