import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np


class ChoiceTable:
    choices_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    indices_choices = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    def encode(self, C):
        x = np.zeros((1, 5))
        for i, c in enumerate(C):
            x[i, self.choices_indices[c]] = 1
        return x

    def decode(self, x):
        choice = np.argmax(x)
        return self.indices_choices[choice]


# Generate dummy data
choice_table = ChoiceTable()

z = choice_table.encode('C')

x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(5, size=(1000, 1)), num_classes=5)

print(y_train)
print(z_train)

x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(5, size=(100, 1)), num_classes=5)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# model.fit(x_train, y_train,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(x_test, y_test, batch_size=128)