import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 채점을 위해 랜덤 시드를 고정하는 코드입니다.
# 정확한 채점을 위해 코드를 변경하지 마세요!
np.random.seed(100)

# 선형 회귀 클래스 구현
class LinearModel:
    def __init__(self):
        # 1. 가중치 초기값을 1.5의 값을 가진 변수 텐서로 설정하세요.
        self.W = tf.Variable(1.5)
        
        # 1. 편향 초기값을 1.5의 값을 가진 변수 텐서로 설정하세요.
        self.b = tf.Variable(1.5)
        
    def __call__(self, X, Y):
        # 2. W, X, b를 사용해 선형 모델을 구현하세요.
        return tf.add(tf.multiply(self.W,X), self.b)
    
# 3. MSE 값으로 정의된 loss 함수 선언 
def loss(y, pred):
    return tf.reduce_mean(tf.square(y-pred))

# gradient descent 방식으로 학습 함수 선언
def train(linear_model, x, y):

    with tf.GradientTape() as t:
        current_loss = loss(y, linear_model(x, y))
    
    # learning_rate 값 선언
    learning_rate = 0.001
    
    # gradient 값 계산
    delta_W, delta_b = t.gradient(current_loss, [linear_model.W, linear_model.b])
    
    # learning rate와 계산한 gradient 값을 이용하여 업데이트할 파라미터 변화 값 계산 
    W_update = (learning_rate * delta_W)
    b_update = (learning_rate * delta_b)
    
    return W_update,b_update
 
def main():
    # 데이터 생성
    x_data = np.linspace(0, 10, 50)
    y_data = 4 * x_data + np.random.randn(*x_data.shape)*4 + 3

    # 데이터 출력
    plt.scatter(x_data,y_data)
    plt.savefig('data.png')
    elice_utils.send_image('data.png')

    # 선형 함수 적용
    linear_model = LinearModel()

    # epochs 값 선언
    epochs = 100

    # epoch 값만큼 모델 학습
    for epoch_count in range(epochs):

        # 선형 모델의 예측 값 저장
        y_pred_data=linear_model(x_data, y_data)

        # 예측 값과 실제 데이터 값과의 loss 함수 값 저장
        real_loss = loss(y_data, linear_model(x_data, y_data))

        # 현재의 선형 모델을 사용하여  loss 값을 줄이는 새로운 파라미터로 갱신할 파라미터 변화 값을 계산
        update_W, update_b = train(linear_model, x_data, y_data)
        
        # 선형 모델의 가중치와 편향을 업데이트합니다. 
        linear_model.W.assign_sub(update_W)
        linear_model.b.assign_sub(update_b)

        # 20번 마다 출력 (조건문 변경 가능)
        if (epoch_count%20==0):
            print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")
            print('W: {}, b: {}'.format(linear_model.W.numpy(), linear_model.b.numpy()))

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(x_data,y_data)
            ax1.plot(x_data,y_pred_data, color='red')
            plt.savefig('prediction.png')
            elice_utils.send_image('prediction.png')
            
if __name__ == "__main__":
    main()