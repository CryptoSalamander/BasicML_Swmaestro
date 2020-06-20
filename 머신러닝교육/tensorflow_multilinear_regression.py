from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random

from data import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# seed를 고정하는 코드입니다.
# 정확한 채점을 위하여 코드를 변경하지 마세요!
tf.random.set_seed(123)
np.random.seed(123)


# advertising.csv 데이터가 X와 Y에 저장됩니다.
#  X는 (200, 3) 의 shape을 가진 2차원 np.array,
#  Y는 (200,) 의 shape을 가진 1차원 np.array 입니다.

#  X는 FB, TV, Newspaper column 에 해당하는 데이터,
#  Y는 Sales column 에 해당하는 데이터가 저장됩니다.
X,Y = read_data()


# 1. 학습용 데이터와 테스트용 데이터로 분리합니다.(80:20)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)



# 2. MSE 값을 1 이하로 낮추는 모델 구현하기
# 2-1. 인공신경망 모델 구성하기
model = tf.keras.models.Sequential([
    # Input Layer
    tf.keras.layers.Dense(20, input_dim = 3, activation='relu'),
    # MSE 값을 1 이하로 낮출 수 있도록 여러 층의 hidden layer를 추가해보세요.
    ##########################################################
    tf.keras.layers.Dense(20, activation='relu'),
    
    ##########################################################
    # Output Layer
    tf.keras.layers.Dense(1)
])

# 모델 학습 방법 설정
model.compile(loss= 'mean_squared_error',
                optimizer='adam')

# 2-2. epochs 값 설정 후 모델 학습 
model.fit(x_train, y_train, epochs=500)


# 학습된 신경망 모델을 사용하여 예측값 생성 및 loss 출력
predicted = model.predict(x_test)

mse_test = mean_squared_error(predicted, y_test)
print("MSE on test data: {}".format(mse_test))
