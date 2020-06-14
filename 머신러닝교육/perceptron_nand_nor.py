import numpy as np
from elice_utils import EliceUtils

elice_utils = EliceUtils()

# 1. NAND_gate 함수를 구현하세요.
def NAND_gate(x1, x2):
    x = np.array([x1,x2])
    weight = np.array([-0.5,-0.5])
    bias = 0.75
    y = np.sum(x*weight)+bias
    
    return Step_Function(y)
    
# 2. NOR gate 함수를 구현하세요.
def NOR_gate(x1, x2):
    x = np.array([x1,x2])
    weight = np.array([-0.5,-0.5])
    bias = 0.1
    y = np.sum(x*weight)+bias
    
    return Step_Function(y) 
    
# 3. Step Function 함수를 구현하세요.
# 앞 실습에서 구현한 함수를 그대로 사용할 수 있습니다.
def Step_Function(y):
    return 1 if y > 0 else 0
    
    
def main():
    
    # NAND, NOR Gate에 넣어줄 Input
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # NAND, NOR Gate를 만족하는지 출력하여 확인
    print('NAND Gate 출력')
    for x1, x2 in array:
        print('Input: ',x1, x2, ' Output: ',NAND_gate(x1, x2))
        
    print('NOR Gate 출력')
    for x1, x2 in array:
        print('Input: ',x1, x2, ' Output: ',NOR_gate(x1, x2))
        

        
if __name__ == "__main__":
    main()