import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Embedding, LSTM, Concatenate
from keras.optimizers import SGD
import numpy as np


class ChoiceTable:
    choices_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    indices_choices = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    def encode(self, C):
        # x = np.zeros((1, 5))
        # for i, c in enumerate(C):
        #     x[i, self.choices_indices[c]] = 1
        # return x
        x = np.zeros((5))
        x[self.choices_indices[C]] = 1
        return x

    def decode(self, x):
        choice = np.argmax(x)
        return self.indices_choices[choice]


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# Generate dummy data
choice_table = ChoiceTable()

z = choice_table.encode('D')

x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(5, size=(1000, 1)), num_classes=5)
z_train = np.array([z for i in range(1000)])

x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(5, size=(100, 1)), num_classes=5)
z_test = np.array([z for i in range(100)])


print(x_train.shape)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
# main_input = Input(shape=(1,), dtype='int32', name='main_input')

# model.add(Embedding(output_dim=32, input_dim=1000, input_length=1))
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
# model.add(Concatenate(main_input))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()

# quit(1)

# model.fit(x_train, z_train,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(x_test, z_test, batch_size=128)


for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              validation_data=(x_test, y_test))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_test))
        rowx, rowy = x_test[np.array([ind])], y_test[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        # q = ctable.decode(rowx[0])
        correct = choice_table.decode(rowy[0])
        guess = choice_table.decode(preds[0])
        # print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)

        # print(correct)