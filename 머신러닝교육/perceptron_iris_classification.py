import sklearn 
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 1. iris data를 읽어 X와 Y에 저장해 반환하는 load_irisdata 함수를 구현하세요.
def load_irisdata():
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    
    
    return X,Y

def main():
    # iris data를 읽어 X와 Y에 저장합니다.
    X,Y = load_irisdata()
    
    # 2. X, Y 데이터를 훈련용 데이터와 테스트 데이터로 분류합니다.(80:20)
    x_train, x_test, y_train, y_test = train_test_split(X,Y,
        test_size = 0.2,
        random_state = 100
    )
    
    
    # 3. sklearn의 퍼셉트론 클래스를 사용하여 train 데이터에 대해 학습하세요.
    perceptron = Perceptron(max_iter=100, eta0=1)
    perceptron.fit(x_train,y_train)
    
    
    #4. test 데이터에 대한 예측값을 생성합니다.
    pred = perceptron.predict(x_test)
    
    print("Test 데이터에 대한 정확도 : %f" %accuracy_score(pred, y_test))
    
    return x_train,x_test,y_train,y_test,pred
    
    
if __name__ == "__main__":
    main()