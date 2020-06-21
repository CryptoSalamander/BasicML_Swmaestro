import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils
elice_utils = EliceUtils()

from mpl_toolkits.mplot3d import Axes3D

# ä���� ���� �ڵ��Դϴ�.
# ��Ȯ�� ä���� ���� �ڵ带 �������� ������!
np.random.seed(100)
tf.random.set_seed(100)

# ������ ����
x_data = np.linspace(0, 10, 100)
y_data = 1.5 * x_data**2 -12 * x_data + np.random.randn(*x_data.shape)*2 + 0.5


# 1. �Ű�� �� ����
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20,input_dim=1,activation='relu'),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1)

])

# 2. �� �н� ��� ����
model.compile(loss='mean_squared_error', optimizer='adam')

# 3. �� �н� 
model.fit(x_data,y_data,epochs=500,verbose=2)

# 4. �н��� ���� ����Ͽ� ������ ���� �� ����
predictions = model.predict(x_data)

# ������ ���
plt.scatter(x_data,y_data)
plt.savefig('data.png')
elice_utils.send_image('data.png')

# ��� ���� �����Ϳ� ������ ���
plt.scatter(x_data,predictions, color='red')
plt.savefig('prediction.png')
elice_utils.send_image('prediction.png')