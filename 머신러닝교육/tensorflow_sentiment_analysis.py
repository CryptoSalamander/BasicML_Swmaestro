import io
import matplotlib as mpl
mpl.use("Agg")

import logging
logging.getLogger('tensorflow').disabled = True

import matplotlib.pyplot as plt
import numpy as np
import re
import math
import tensorflow as tf
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.feature_extraction.text import CountVectorizer

import elice_utils
elice_utils = elice_utils.EliceUtils()


# seed를 고정하는 코드입니다.
# 정확한 채점을 위하여 값을 변경하지 마세요!
tf.random.set_seed(123)
np.random.seed(123)

special_chars_remover = re.compile("[^\w'|_]")

# 특수 문자를 제거하는 함수입니다.
def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)

# 1. /data/ratings.txt 에서 데이터를 읽어, 인공신경망 학습을 위한 두 개의 리스트를 반환합니다.
def read_data():
    sentences = []
    labels = []
    
    with open("data/ratings.txt") as fp:
        next(fp)
        for line in fp:
            dp = line.split("\t")
            sentences.append(remove_special_characters(dp[1]))
            labels.append(int(dp[2]))
    
    return sentences,labels

# 2. count_vect 함수를 완성하세요.
def count_vect(sentences, testing_sentence):
    
    # 테스트 문장 또한 토큰 빈도수 안에 포함되어야하기 때문에 sentences 리스트에 추가합니다. 
    sentences.append(testing_sentence)
    
    # sentences를 카운트 벡터로 변환하세요.
    Vectorizer = CountVectorizer(min_df=1)
    vector = Vectorizer.fit_transform(sentences)
    vector = vector.toarray()
    
    return vector

# ANN 함수를 완성하세요.
def ANN(vector,labels):
    
    # 카운트 벡터로 변환된 테스트 문장 벡터를 저장합니다.
    test = vector[-1]
    # 모델 학습 데이터에서 테스트 데이터를 제거합니다.
    vector = vector[:-1]
    # 모델 입력을 위한 형태로 변환합니다.
    test = [[test]]
    
    # 입력 데이터의 차원은 카운트 벡터 안의 토큰 수 입니다.
    num_voca = len(vector[0])
    
    # 인공 신경망 생성
    ANN_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, input_dim = num_voca, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
    ])

    # 3. loss와 optimizer를 설정하세요.
    ANN_model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam')

    # 학습 시작 
    ANN_model.fit(vector, labels , epochs=500, verbose=0)
    
    predict = ANN_model.predict(test)
    
    return predict

def main():
    
    train_sentences,labels = read_data()
    
    testing_sentence = "어설픈 연기들로 몰입이 전혀 안되네요"
        
    bow_vect = count_vect(train_sentences, testing_sentence)
    probs = ANN(bow_vect,labels)
    
    # 시각화 코드입니다.
    plot_title = testing_sentence
    if len(plot_title) > 50: plot_title = plot_title[:50] + "..."
    visualize_boxplot(plot_title,
                  [probs[0][0],probs[0][1]],
                  ['Negative', 'Positive'])

def visualize_boxplot(title, values, labels):
    width = .35

    print(title)
    
    fig, ax = plt.subplots()
    ind = np.arange(len(values))
    rects = ax.bar(ind, values, width)
    ax.bar(ind, values, width=width)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(labels)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., height + 0.01, '%.2lf%%' % (height * 100), ha='center', va='bottom')

    autolabel(rects)

    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")

if __name__ == "__main__":
    main()
