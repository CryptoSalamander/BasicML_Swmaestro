import numpy as np
from elice_utils import EliceUtils

elice_utils = EliceUtils()

# 1. `AND_gate` �Լ��� �����ϼ���. 
def AND_gate(x1,x2):
    x = np.array([x1,x2])
    weight = np.array([0.5,0.5])
    bias = -0.75
    
    return Step_Function(np.sum(x*weight) + bias)
    
# 2. `OR_gate` �Լ��� �����ϼ���.
def OR_gate(x1,x2):
    x = np.array([x1,x2])
    weight = np.array([0.5,0.5])
    bias = -0.1
    
    return Step_Function(np.sum(x*weight) + bias)
    
# 3. `NAND_gate` �Լ��� �����ϼ���.
def NAND_gate(x1,x2):
    x = np.array([x1,x2])
    weight = np.array([-0.5,-0.5])
    bias = 0.75
    
    return Step_Function(np.sum(x*weight) + bias)
    
# 4. Step_Function �Լ��� �����ϼ���.
def Step_Function(y):
    return 1 if y > 0 else 0
    
# 5. ������ AND, OR, NAND gate �Լ����� Ȱ���Ͽ� XOR_gate �Լ��� �����ϼ���. 
def XOR_gate(x1, x2):
    A = NAND_gate(x1,x2)
    B = OR_gate(x2,x1)
    Q = AND_gate(A,B)
    y = Q
    
    return Q
    

def main():
    # NOR gate�� �־��� Input
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # XOR gate�� �����ϴ��� ����Ͽ� Ȯ��
    print('XOR Gate ���')
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ', XOR_gate(x1, x2))


if __name__ == "__main__":
    main()