"""
主題 04-1. 用RNN做情意分析
我們終於要介紹三大神經網路的最後一個, 也就是 RNN。RNN 有不少的變型, 例如 LSTM 和 GRU 等等, 不過我們都通稱叫 RNN。RNN 是一種「有記憶」的神經網路, 非常適合時間序列啦, 或是不定長度的輸入資料。

我們來看看怎麼樣用 RNN 做電影評論的「情意分析」, 也就是知道一則評論究竟是「正評」還是「負評」。

1. 初始準備
基本上和之前是一樣的, 我們就不再說明。
"""

#%env KERAS_BACKEND=tensorflow
#env: KERAS_BACKEND=tensorflow

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
"""
2. 讀入 IMDB 電影數據庫
今天我們要評入 IMDB 電影數據庫影評的部份。
"""

from keras.datasets import imdb
#Using TensorFlow backend.


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
#要注意這裡我們限制只選「最常用」1 萬字, 也就是超過這範圍的就當不存在。這是文字分析常會做的事。


print("訓練資料總筆數 =", len(x_train))
print("測試資料總筆數 =", len(x_test))
訓練資料總筆數 = 25000
測試資料總筆數 = 25000
"""
2.1 輸入資料部份
我們來看一下輸入部份長什麼樣子?
"""

x_train[99]

#注意這其實是一個 list 而不是 array, 原因是每筆資料 (每段影評) 長度自然是不一樣的! 我們檢查一下前 10 筆的長度就可以知道。


type(x_train[99])
list

for i in range(10):
    print(len(x_train[i]), end=', ')
"""
最後要說明的是, 在每筆輸入資料的數字都代表英文的一個單字。編號方式是在我們資料庫裡所有文字的排序: 也就是出現頻率越高, 代表的數字就越小。

2.2 輸出資料部份
輸出方面應該很容易想像, 我們來看看前 10 筆。結果自然就是 0 (負評) 或 1 (正評)。
"""

y_train[:10]
array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])
"""
2.3 送入神經網路的輸入處理
雖然 RNN 是可以處理不同長度的輸入, 在寫程式時我們還是要

設輸入文字長度的上限
把每段文字都弄成一樣長, 太短的後面補上 0
"""
from keras.preprocessing import sequence

x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

x_train.shape
(25000, 100)

x_train[99]

"""
至此我們可以來寫我們的第一個 RNN 了!

3. 打造你的 RNN
這裡我們選用 LSTM, 基本上用哪種 RNN 寫法都是差不多的!

3.1 決定神經網路架構
先將 10000 維的文字壓到 128 維
然後用 128 個 LSTM
最後一個 output, 直接用 sigmoid 送出
3.2 建構我們的神經網路
文字我們用 1-hot 表示是很標準的方式, 不過要注意的是, 因為我們指定要 1 萬個字, 所以每個字是用 1 萬維的向量表示! 這一來很浪費記憶空間, 二來字和字間基本上是沒有關係的。我們可以用某種「合理」的方式, 把字壓到比較小的維度, 這些向量又代表某些意思 (比如說兩個字代表的向量角度小表相關程度大) 等等。

這聽來很複雜的事叫 "word embedding", 而事實上 Keras 會幫我們做。我們只需告訴 Keras 原來最大的數字是多少 (10000), 還有我們打算壓到幾維 (128)。
"""

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

model = Sequential()

model.add(Embedding(10000, 128))
#LSTM 層, 我們做 150 個 LSTM Cells。


model.add(LSTM(150))
#單純透過 sigmoid 輸出。


model.add(Dense(1, activation='sigmoid'))
"""
3.3 組裝
這次我們用 binary_crossentropy 做我們的 loss function, 另外用一個很潮的 Adam 學習法。
"""

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
"""
4. 訓練
我們用的 embedding 中, 會被 batch_size 影響輸入。輸入的 shape 會是

(batch_size, 每筆上限)
也就是 (32,100) 輸出是 (32,100,128), 其中 128 是我們決定要壓成幾維的向量。
"""

model.fit(x_train, y_train, 
          batch_size=32, 
          epochs=15)
"""
5. 檢視結果
5.1 分數
我們照例來看看測試資料的分數。
"""

score = model.evaluate(x_test, y_test)


print('測試資料的 loss:', score[0])
print('測試資料正確率:', score[1])

"""
5.2 儲存結果
這裡有 8 成我們可以正確分辨, 看來還不差, 照例我們把結果存檔。
"""