import numpy as np
from elice_utils import EliceUtils

elice_utils = EliceUtils()

# 1. NAND_gate �Լ��� �����ϼ���.
def NAND_gate(x1, x2):
    x = np.array([x1,x2])
    weight = np.array([-0.5,-0.5])
    bias = 0.75
    y = np.sum(x*weight)+bias
    
    return Step_Function(y)
    
# 2. NOR gate �Լ��� �����ϼ���.
def NOR_gate(x1, x2):
    x = np.array([x1,x2])
    weight = np.array([-0.5,-0.5])
    bias = 0.1
    y = np.sum(x*weight)+bias
    
    return Step_Function(y) 
    
# 3. Step Function �Լ��� �����ϼ���.
# �� �ǽ����� ������ �Լ��� �״�� ����� �� �ֽ��ϴ�.
def Step_Function(y):
    return 1 if y > 0 else 0
    
    
def main():
    
    # NAND, NOR Gate�� �־��� Input
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # NAND, NOR Gate�� �����ϴ��� ����Ͽ� Ȯ��
    print('NAND Gate ���')
    for x1, x2 in array:
        print('Input: ',x1, x2, ' Output: ',NAND_gate(x1, x2))
        
    print('NOR Gate ���')
    for x1, x2 in array:
        print('Input: ',x1, x2, ' Output: ',NOR_gate(x1, x2))
        

        
if __name__ == "__main__":
    main()