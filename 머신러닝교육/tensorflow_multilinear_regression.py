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

# seed�� �����ϴ� �ڵ��Դϴ�.
# ��Ȯ�� ä���� ���Ͽ� �ڵ带 �������� ������!
tf.random.set_seed(123)
np.random.seed(123)


# advertising.csv �����Ͱ� X�� Y�� ����˴ϴ�.
#  X�� (200, 3) �� shape�� ���� 2���� np.array,
#  Y�� (200,) �� shape�� ���� 1���� np.array �Դϴ�.

#  X�� FB, TV, Newspaper column �� �ش��ϴ� ������,
#  Y�� Sales column �� �ش��ϴ� �����Ͱ� ����˴ϴ�.
X,Y = read_data()


# 1. �н��� �����Ϳ� �׽�Ʈ�� �����ͷ� �и��մϴ�.(80:20)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)



# 2. MSE ���� 1 ���Ϸ� ���ߴ� �� �����ϱ�
# 2-1. �ΰ��Ű�� �� �����ϱ�
model = tf.keras.models.Sequential([
    # Input Layer
    tf.keras.layers.Dense(20, input_dim = 3, activation='relu'),
    # MSE ���� 1 ���Ϸ� ���� �� �ֵ��� ���� ���� hidden layer�� �߰��غ�����.
    ##########################################################
    tf.keras.layers.Dense(20, activation='relu'),
    
    ##########################################################
    # Output Layer
    tf.keras.layers.Dense(1)
])

# �� �н� ��� ����
model.compile(loss= 'mean_squared_error',
                optimizer='adam')

# 2-2. epochs �� ���� �� �� �н� 
model.fit(x_train, y_train, epochs=500)


# �н��� �Ű�� ���� ����Ͽ� ������ ���� �� loss ���
predicted = model.predict(x_test)

mse_test = mean_squared_error(predicted, y_test)
print("MSE on test data: {}".format(mse_test))
