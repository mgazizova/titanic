import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras
import model_plot
import seaborn as sns
from keras.optimizers import SGD


#far too much repetition, just cleaning the data for the neural net
df = pd.read_csv('titanic.csv')
control = pd.read_csv('test.csv')
input_data = pd.read_csv('train.csv')



train, test = train_test_split(input_data, test_size=0.2)   # тренировочные - 80%   тестовые - 20%
train = train.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)  # убрали Имя, Номер кабины, Билет, Идентификатор
train = train.dropna()  # обрезали записи, где нет информации
test = test.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)    # убрали Имя, Номер кабины, Билет, Идентификатор
test = test.dropna()    # обрезали записи, где нет информации
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X_train = train.drop(['Survived'], 1)
X_train = X_train.as_matrix()
X_test = test.drop(['Survived'], 1)
X_test = X_test.as_matrix()
y_train = train['Survived']
y_test = test['Survived']

# from class vector to binary matrix
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

nb_classes = 2
nb_epoch = 100
batch_size = 32

model = Sequential()
model.add(Dense(32, input_shape=X_train.shape[1:], activation='relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), verbose=2)

# Plot training & validation accuracy values
model_plot.accuracy_plot(history)

# Plot training & validation loss values
model_plot.loss_plot(history)
