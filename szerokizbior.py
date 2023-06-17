import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Załadowanie zbioru danych IMDb
max_features = 20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Wyrównanie długość sekwencji i przekonwertowanie ich na wektory o stałej długości
maxlen = 80
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Budowanie modelu sieci neuronowej
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trening modelu na zbiorze danych treningowych
batch_size = 32
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Utworzenie klasyfikacji recenzji
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test accuracy:', score[1])
