%env KERAS_BACKEND=tensorflow

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

len(x_train)

len(x_test)

x_train[9487].shape

plt.imshow(x_train[9487], cmap='Greys')

y_train[9487]

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


y_train[9487]


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(500, input_dim=784))
model.add(Activation('sigmoid'))

model.add(Dense(500))
model.add(Activation('sigmoid'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])

model.summary()

from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='model01.png')

model.fit(x_train, y_train, batch_size=100, epochs=20)

from ipywidgets import interact_manual
predict = model.predict_classes(x_test)


def test(測試編號):
    plt.imshow(x_test[測試編號].reshape(28,28), cmap="Greys")
    print("神經網路判斷為:", predict[測試編號])


interact_manual(test, 測試編號 = (0, 9999));

score = model.evaluate(x_test, y_test)

print('測試資料的 loss:', score[0])
print('測試資料正確率:', score[1])

model_json = model.to_json()
open('handwriting_model_architecture.json', 'w').write(model_json)
model.save_weights('handwriting_model_weights.h5')




