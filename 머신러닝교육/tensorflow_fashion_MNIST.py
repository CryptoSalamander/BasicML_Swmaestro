from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import elice_utils
eu = elice_utils.EliceUtils()

# seed�� �����ϴ� �ڵ��Դϴ�.
# ��Ȯ�� ä���� ���Ͽ� ���� �������� ������!
np.random.seed(100)
tf.random.set_seed(100)

def ANN_classifier(x_train, y_train):

    # 1-1. �ΰ� �Ű�� �з� ���� �����մϴ�.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    
    # 1-2. ���� �н��� loss�� optimizer�� �����մϴ�.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # 1-3. ���� �н��� epochs ���� �����մϴ�.
    model.fit(x_train, y_train, epochs=100)

    return model
    
def main():
    
    x_train = np.loadtxt('./data/train_images.csv', delimiter =',', dtype = np.float32)
    y_train = np.loadtxt('./data/train_labels.csv', delimiter =',', dtype = np.float32)
    x_test = np.loadtxt('./data/test_images.csv', delimiter =',', dtype = np.float32)
    y_test = np.loadtxt('./data/test_labels.csv', delimiter =',', dtype = np.float32)

    
    # �̹��� �����͸� 0~1������ ������ �ٲپ� �ݴϴ�.
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = ANN_classifier(x_train,y_train)
    
    # �н��� ���� test �����͸� Ȱ���Ͽ� ���մϴ�.
    loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print('\n- TEST ��Ȯ�� :', test_acc)
    
    # ������ 3���� test data�� �̹����� ���̺��� ����ϰ� ������ ���̺� ���
    predictions = model.predict(x_test)
    rand_n = np.random.randint(100, size=3)

    for i in rand_n:
        img = x_test[i].reshape(28,28)
        plt.imshow(img,cmap="gray")
        plt.show()
        plt.savefig("test.png")
        eu.send_image("test.png")

        print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions[i]))
        
    
if __name__ == "__main__":
    main()