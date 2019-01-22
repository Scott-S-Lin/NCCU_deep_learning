""" 
 主題 02-1. 用 CNN 做圖形辨識
Yann LeCun 被譽為 Deep Learning 的三巨頭之一。他的 CNN (Convolutional Neural Networks) 是讓 Neural Network 重新受到重視的主因之一。

1. 初始準備
基本上和之前是一樣的, 我們就不再說明。


%env KERAS_BACKEND=tensorflow
env: KERAS_BACKEND=tensorflow

%matplotlib inline
"""

import numpy as np
import matplotlib.pyplot as plt

"""
2. 讀入 MNIST 數據庫
2.1 由 Keras 讀入 MNIST
基本上和我們上次一樣, 這次因為 Keras 已偷偷把數據庫存在你的電腦, 所以會快很多!
"""

from keras.datasets import mnist
#Using TensorFlow backend.


(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
2.2 輸入格式整理
如果你還記得, 我們每筆輸入資料都是 28x28 的陣列, CNN 其實就是吃「圖」的, 所以基本上不用像之前把每筆資料拉平。「但。是。」平常的圖都有 R, G, B 三個 channels, 每個 channel 都是一個矩陣, 也就是一張圖可能是三個矩陣! 我們是灰階, 也就是只有一個 channel。但這件事也要明確的告訴 Keras。

換句話說, 我們的輸入每筆資料型式要從 (28, 28) 換成 (28, 28, 1)!
"""

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#確認一下...


x_train[9487].shape


x_train[9487][:,:,0]

"""
2.3 輸出格式整理
和上次一樣, 我們用標準 1-hot 方式處理。
"""

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
"""
3. 打造你的 CNN
3.1 決定神經網路架構、讀入相關套件
CNN 我們一樣要決定用幾層的 CNN, 然後是不是每次都要做 max-pooling。再來就是拉平、送入標準神經網路 (再度要決定幾層、幾個神經元)。

做 3 次 convolution, 每次都接 max-pooling
filter 大小都是 3x3, max-pooling 都用 2x2 為一小區塊
CNN 一個小技巧是每層的 filters 數目是越來越多。做完 convolution 之後, 我們要拉平、再送入一個標準的神經網路。這個神經網路設計是這樣:

只有 1 個隱藏層, 使用 200 個神經元。
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
"""
3.2 建構我們的神經網路
一開始一樣是打開個空白的神經網路。
"""

model = Sequential()
#第一個隱藏層一樣要告訴 Keras 我們輸入長什麼樣子。padding 設成 same 是每個 filter 會輸出原來 28x28 一樣大小的矩陣。


model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28,28,1)))
model.add(Activation('relu'))
#Max-Pooling!


model.add(MaxPooling2D(pool_size=(2, 2)))
#第二次 Convolution!


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
#再 Max-Pooling!


model.add(MaxPooling2D(pool_size=(2, 2)))
#第三次 Convolution!


model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
#Max-Pooling 最終回。


model.add(MaxPooling2D(pool_size=(2, 2)))
#然後我們要送進一般的神經網路了。記得這是要拉平的, 還在 Keras 會幫我們做!


model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
#輸出和上次一樣!


model.add(Dense(10))
model.add(Activation('softmax'))
"""
3.3 組裝
和之前比較不一樣的是我們還要做 compile 才正式把我們的神經網路建好。
"""

#model.compile(loss="categorical_crossentropy",
#              optimizer=Adadelta(),
#              metrics=['accuracy'])

model.compile(loss='mse', optimizer=SGD(lr=0.05), metrics=['accuracy'])

#3.4 檢視我們的神經網路

model.summary()

#4. 訓練

model.fit(x_train, y_train, batch_size=100, epochs=12)
"""
5. 結果測試
5.1 分數
"""
score = model.evaluate(x_test, y_test)


print('測試資料的 loss:', score[0])
print('測試資料正確率:', score[1])
#測試資料的 loss: 0.00221789917492
#測試資料正確率: 0.9853
"""
5.2 儲存結果
結果看來還不差, 所以我們把結果存起來。
"""

model_json = model.to_json()
open('handwriting_model_cnn.json', 'w').write(model_json)
model.save_weights('handwriting_weights_cnn.h5')
"""
5.3 欣賞一下成果
我們用另一個方式: 每次選 5 個顯示, 看是不是有正確辨識。
"""

predict = model.predict_classes(x_test)

pick = np.random.randint(1,9999, 5)


for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[pick[i]].reshape(28,28), cmap='Greys')
    plt.title(predict[pick[i]])
    plt.axis("off")

"""
5.4 小結論
我們到此, 基本上是「亂做」的神經網路。有些同學在不斷試驗的過程中, 可能會發現有時會出現很糟糕的結果。因此, 接下來我們要介紹怎麼樣用些簡單的手法, 能讓學習效果比較穩定, 而且有可能可以增加學習效率。
"""
