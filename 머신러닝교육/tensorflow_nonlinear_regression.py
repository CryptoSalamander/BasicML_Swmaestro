import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils
elice_utils = EliceUtils()

from mpl_toolkits.mplot3d import Axes3D

# 채점을 위한 코드입니다.
# 정확한 채점을 위해 코드를 수정하지 마세요!
np.random.seed(100)
tf.random.set_seed(100)

# 데이터 생성
x_data = np.linspace(0, 10, 100)
y_data = 1.5 * x_data**2 -12 * x_data + np.random.randn(*x_data.shape)*2 + 0.5


# 1. 신경망 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20,input_dim=1,activation='relu'),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1)

])

# 2. 모델 학습 방법 설정
model.compile(loss='mean_squared_error', optimizer='adam')

# 3. 모델 학습 
model.fit(x_data,y_data,epochs=500,verbose=2)

# 4. 학습된 모델을 사용하여 예측값 생성 및 저장
predictions = model.predict(x_data)

# 데이터 출력
plt.scatter(x_data,y_data)
plt.savefig('data.png')
elice_utils.send_image('data.png')

# 곡선형 분포 데이터와 예측값 출력
plt.scatter(x_data,predictions, color='red')
plt.savefig('prediction.png')
elice_utils.send_image('prediction.png')