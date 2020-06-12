from elice_utils import EliceUtils
elice_utils = EliceUtils()

import numpy as np

# 1. AND gate 함수를 구현하세요.
def AND_gate(x1, x2):
    x = np.array([x1, x2])
    
    # x1과 x2에 각각 곱해줄 가중치 0.5, 0.5로 설정
    weight = np.array([0.5,0.5])
    
    # 1-1. AND gate를 만족하는 bias를 설정합니다.
    bias = -0.75
    
    # 1-2. 가중치, 입력값, 편향을 이용하여 가중 신호의 총합을 구합니다.
    y = np.sum(x * weight) + bias
    
    # Step Function 함수를 호출하여 AND gate의 출력값을 반환합니다.
    return Step_Function(y)

# 2. OR gate 함수를 구현하세요.
def OR_gate(x1, x2):
    x = np.array([x1, x2])
    
    # x1과 x2에 각각 곱해줄 가중치 0.5, 0.5로 설정
    weight = np.array([0.5,0.5])
    
    # 2-1. OR gate를 만족하는 bias를 설정합니다.
    bias = 0
    
    # 2-2. 가중치, 입력값, 편향을 이용하여 가중 신호의 총합을 구합니다.
    y = np.sum(x * weight) + bias
    
    #Step Function 함수를 호출하여 AND gate의 출력값을 반환합니다.
    return Step_Function(y)

# 3. Step Function 구현
def Step_Function(y):
    if y > 0:
        return 1
    else:
        return 0   
    
def main():
    
    # AND Gate와 OR Gate에 넣어줄 Input 입니다.
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # AND Gate를 만족하는지 출력하여 확인합니다.
    print('AND Gate 출력')
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ',AND_gate(x1, x2))
        
    # OR Gate를 만족하는지 출력하여 확인합니다.
    print('\nOR Gate 출력')
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ',OR_gate(x1, x2))

if __name__ == "__main__":
    main()