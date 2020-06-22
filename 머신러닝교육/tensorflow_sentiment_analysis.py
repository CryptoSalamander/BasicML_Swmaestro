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


# seed�� �����ϴ� �ڵ��Դϴ�.
# ��Ȯ�� ä���� ���Ͽ� ���� �������� ������!
tf.random.set_seed(123)
np.random.seed(123)

special_chars_remover = re.compile("[^\w'|_]")

# Ư�� ���ڸ� �����ϴ� �Լ��Դϴ�.
def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)

# 1. /data/ratings.txt ���� �����͸� �о�, �ΰ��Ű�� �н��� ���� �� ���� ����Ʈ�� ��ȯ�մϴ�.
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

# 2. count_vect �Լ��� �ϼ��ϼ���.
def count_vect(sentences, testing_sentence):
    
    # �׽�Ʈ ���� ���� ��ū �󵵼� �ȿ� ���ԵǾ���ϱ� ������ sentences ����Ʈ�� �߰��մϴ�. 
    sentences.append(testing_sentence)
    
    # sentences�� ī��Ʈ ���ͷ� ��ȯ�ϼ���.
    Vectorizer = CountVectorizer(min_df=1)
    vector = Vectorizer.fit_transform(sentences)
    vector = vector.toarray()
    
    return vector

# ANN �Լ��� �ϼ��ϼ���.
def ANN(vector,labels):
    
    # ī��Ʈ ���ͷ� ��ȯ�� �׽�Ʈ ���� ���͸� �����մϴ�.
    test = vector[-1]
    # �� �н� �����Ϳ��� �׽�Ʈ �����͸� �����մϴ�.
    vector = vector[:-1]
    # �� �Է��� ���� ���·� ��ȯ�մϴ�.
    test = [[test]]
    
    # �Է� �������� ������ ī��Ʈ ���� ���� ��ū �� �Դϴ�.
    num_voca = len(vector[0])
    
    # �ΰ� �Ű�� ����
    ANN_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, input_dim = num_voca, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
    ])

    # 3. loss�� optimizer�� �����ϼ���.
    ANN_model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam')

    # �н� ���� 
    ANN_model.fit(vector, labels , epochs=500, verbose=0)
    
    predict = ANN_model.predict(test)
    
    return predict

def main():
    
    train_sentences,labels = read_data()
    
    testing_sentence = "��� ������ ������ ���� �ȵǳ׿�"
        
    bow_vect = count_vect(train_sentences, testing_sentence)
    probs = ANN(bow_vect,labels)
    
    # �ð�ȭ �ڵ��Դϴ�.
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
