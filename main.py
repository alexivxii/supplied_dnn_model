import tensorflow as tf
from tensorflow import keras

import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.losses import MeanSquaredError
from keras.optimizers import Adam, Nadam
from matplotlib import pyplot
import statistics
from keras.models import load_model

groceries_list = {
    1: "milk",
    2: "bread",
    3: "eggs",
    4: "cheese",
    5: "yogurt",
    6: "butter",
    7: "orange juice",
    8: "apple juice",
    9: "soda",
    10: "water",
    11: "beer",
    12: "wine",
    13: "chips",
    14: "cookies",
    15: "crackers",
    }

#dataset_size - numarul de seturi de saptamani (nr de persoane)
dataset_size = 5000
#weeks_per_person - numarul de saptamani pe fiecare persoana
weeks_per_person = 4

x = [] # train input data
y = [] # train output data

for person in range(dataset_size):
    products_bought = []  # lista in care adaugam valori de 0 sau 1, respectiv daca produsul a fost adaugat sau nu in lista de cumparaturi, din lista de produse posibile (pe toate saptamanile weeks_per_person)
    bought_5th_week = [0 for item in
                       groceries_list]  # prezicem a 5-a saptamana, len(bought_5th_week) = len(groceries_list)
    for week in range(weeks_per_person):
        bought = [random.randint(0, 1) for product in
                  groceries_list]  # lista in care adaugam valori de 0 sau 1, respectiv daca produsul a fost adaugat sau nu in lista de cumparaturi, din lista de produse posibile (pe o singura saptamana)

        for index in range(len(bought)):
            bought_5th_week[index] += bought[index]  # adaugam valorile de 0,1 pentru ca apoi sa facem average

        products_bought += bought

    # average pentru valorile din a 5-a saptamana
    for index in range(len(bought_5th_week)):
        bought_5th_week[index] /= weeks_per_person
        bought_5th_week[index] = round(bought_5th_week[index])

    # bought_5th_week = [random.randint(0, 1) for product in groceries_list]

    x.append(products_bought)
    y.append(bought_5th_week)

# print(len(products_bought))


x = np.array(x)
print(x.shape)
# x.shape[0] should be the number of people in the dataset ( = dataset_size )
# x.shape[1] should be the number of total products*number of weeks_per_person ( len(groceries_list)*weeks_per_person )

y = np.array(y)
print(y.shape)
# y.shape[1] should be of the same size as the groceries list


#impartirea datelor pentru antrenare,testare si validare

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# setul de date de validare e un set de test intermediar, dupa fiecare etapa de training.
# intrucat metoda .fit() din Keras imparte singura datele de input in _train si _validare, nu mai e nevoie sa impartim noi

print(x_train.shape)
print(x_test.shape)
# print(x_val.shape)
print(y_train.shape)
print(y_test.shape)
# print(y_val.shape)


print(len(groceries_list))


def get_model():
  # definim arhitectua modelului

  model = Sequential()
  model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(len(groceries_list), activation='linear')) # len(groceries_list) initial este 15

  return model


def show_overfit(history, metricName='mse', lossName='loss'):

  acc = history.history[metricName]
  val_acc = history.history['val_' + metricName]
  loss = history.history[lossName]
  val_loss = history.history['val_' + lossName]
  epochs = range(1, len(acc) + 1)
  pyplot.plot(epochs, acc, 'bo', label='Training ' + str(metricName))
  pyplot.plot(epochs, val_acc, 'b', label='Validation ' + str(metricName))
  pyplot.title('Training and validation ' + str(metricName))
  pyplot.legend()


  pyplot.figure()

  pyplot.plot(epochs, loss, 'bo', label='Training loss')
  pyplot.plot(epochs, val_loss, 'b', label='Validation loss')
  pyplot.title('Training and validation loss')
  pyplot.legend()

  pyplot.show()

  def get_new_compiled_model():
      model = get_model()

      loss = MeanSquaredError()
      opt = Adam()

      model.compile(optimizer=opt, loss=loss, metrics=['mae'])  #

      return model

  def fit_model(model, X_train, y_train):
      batch_size = 32
      epochs = 100

      # checkpoint_filepath = _CHECKPOINT_FILEPATH
      # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      #     filepath=checkpoint_filepath,
      #     save_weights_only=True,
      #     mode='min',
      #     save_best_only=True)

      history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15,
                          verbose=True)  # , callbacks=[model_checkpoint_callback])

      return history

  model = get_new_compiled_model()
  # model.summary()

  history = fit_model(model, x_train, y_train)

  show_overfit(history, metricName='mae')

  loss, mae = model.evaluate(x_test, y_test, verbose=True)
  # loss e definit mai sus in get_new_compiled_model (= MSE)
  # mae = Mean Average Error, alta metrica de evaluare


  print(loss, mae)

  model.save('my_model.h5')
  # salvez modelul antrenat

  loaded_model = load_model('my_model.h5')
  # sintaxa de incarcare a unui model salvat


  ############################### Test pe date separat

  # test_v = np.array([random.randint(0, 1) for i in range(weeks_per_person * len(groceries_list))])
  # test_v = test_v.reshape(1, 60)
  #
  # print(test_v.shape)
  #
  # predictions = loaded_model.predict(test_v).round()
  #
  # print(test_v)
  #
  # for i in range(weeks_per_person):
  #     print(test_v[0][(i * len(groceries_list)):((i + 1) * len(groceries_list))])
  #
  #     print(predictions[0])