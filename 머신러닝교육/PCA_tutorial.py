import csv
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
import elice_utils

def main():
    X, attributes = input_data()
    pca_array = normalize(X)
    pca, pca_array = run_PCA(X, 2)
    visualize_2d_wine(pca_array)

def input_data():
    X = []
    attributes = []
    with open('data/attributes.txt') as fp:
        attributes = fp.readlines()
    attributes = [x.strip() for x in attributes]
    
    csvreader = csv.reader(open("data/wine.csv"))
    for line in csvreader:
        float_numbers = [float(x) for x in line]
        X.append(float_numbers)

    return np.array(X), attributes

def run_PCA(X, num_components):
    pca = sklearn.decomposition.PCA(n_components=num_components)
    pca.fit(X)
    pca_array = pca.transform(X)
    return pca, pca_array
    
def normalize(X):
    '''
    ������ feature�� ����,
    178���� �����Ϳ� ��Ÿ���� �ش��ϴ� feature�� ���� �ּ� 0, �ִ� 1�� �ǵ���
    ���������� �����͸� �̵���ŵ�ϴ�.
    '''
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - np.min(X[:,i])
        X[:,i] = X[:,i] / np.max(X[:,i])

    return X

def visualize_2d_wine(X):
    '''X�� �ð�ȭ�ϴ� �ڵ带 �����մϴ�.'''
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("image.png")
    elice_utils.send_image("image.png")

if __name__ == '__main__':
    main()