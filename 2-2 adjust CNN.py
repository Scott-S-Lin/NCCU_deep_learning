"""
主題 02-2. 作業說明: 調校你的 CNN
說實在我們 CNN 不算太好, 我們可以用幾個技巧, 就可以改善穩定度和學習成效。我們建議這次先不要改變結構 (例如用的 filter 個數等)。

1. 初始準備
基本上和之前是一樣的, 我們就不再說明。
"""

%env KERAS_BACKEND=tensorflow
env: KERAS_BACKEND=tensorflow

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
"""
2. 讀入 MNIST 數據庫
這還是一樣, 然後大家記得可以做 nomarlization!
"""

from keras.datasets import mnist
Using TensorFlow backend.
/Users/mac/anaconda/envs/tensorflow/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255
# 然後的 reshape 等等是一樣的。


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
""""
3. 改變 loss function
大家記得 loss function 就是看我們和目標差多遠。我們最常用的「平均平方差」並不一定是唯一、甚至「最好」的。分類問題我們更常用 "categorical crossentropy", 在 Keras 中要使用很方便, 就這樣下指令:
""""
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])


""""
4. 我們來做 Regularization!
任何函數的學習, 從迴歸、機器學習到神經網路, 都害怕 "overfitting"。所謂 overfitting 大概可以想成同學把考古題做得太熟、差不多都背下來, 但一碰到新的題目就不會!

4.1 Regularization 的兩大手法
為了要避免這件事發生, 我們基本手法大致有兩種, 我們大概選一種用就可以了。

(1) Dropout
Dropout 方式很有趣: 我們設一定的「休息比例」, 也就是每個 batch 讓一些神經元休息, 這樣就不會有「某些神經元拼命背答案」這樣的事發生。實作上其實是設某個比例, 讓有些神經元的輸出不要傳到下一層, 這樣就好像那些神經元 (暫時) 不存在。

(2) Weights 不要太大
這想法很簡單, 如果 weights 太大就「處罰」我們的神經網路! 等等, 怎麼處罰呢? 其實就是把某一層 weight 的大小送進我們的 loss function, 而我們的目標是要讓 loss function 的值最小, 自然達到我們的目的。

5. 不要忘了還可換 Optimizer
6. 恭喜, 你已是神經網路調校大師!¶
我們來總結一下怎麼去調我們的神經網路:

調整神經網路的結構
把數據做 normalization
改變你的 optimizer
選用不同的 loss function
用 regularization 防止 overfitting
千萬千萬不要忘記, 也許你該「重新問你的問題」!

我們還有最後還有兩個重點在課程最後會介紹:

選用適當的 weights 初始化
用 SELU 或是 Bach Normalization 讓過程中都是 normalized 的狀態
""""