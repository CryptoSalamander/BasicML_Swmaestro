import sklearn 
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 1. iris data�� �о� X�� Y�� ������ ��ȯ�ϴ� load_irisdata �Լ��� �����ϼ���.
def load_irisdata():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    
    
    return X,Y

def main():
    # iris data�� �о� X�� Y�� �����մϴ�.
    X,Y = load_irisdata()
    
    # 2. X, Y �����͸� �Ʒÿ� �����Ϳ� �׽�Ʈ �����ͷ� �з��մϴ�.(80:20)
    x_train, x_test, y_train, y_test = train_test_split(X,Y,
        test_size = 0.2,
        random_state = 100
    )
    
    
    # 3. sklearn�� �ۼ�Ʈ�� Ŭ������ ����Ͽ� train �����Ϳ� ���� �н��ϼ���.
    perceptron = Perceptron(max_iter=100, eta0=1)
    perceptron.fit(x_train,y_train)
    
    
    #4. test �����Ϳ� ���� �������� �����մϴ�.
    pred = perceptron.predict(x_test)
    
    print("Test �����Ϳ� ���� ��Ȯ�� : %f" %accuracy_score(pred, y_test))
    
    return x_train,x_test,y_train,y_test,pred
    
    
if __name__ == "__main__":
    main()