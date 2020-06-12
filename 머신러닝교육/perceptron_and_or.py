from elice_utils import EliceUtils
elice_utils = EliceUtils()

import numpy as np

# 1. AND gate �Լ��� �����ϼ���.
def AND_gate(x1, x2):
    x = np.array([x1, x2])
    
    # x1�� x2�� ���� ������ ����ġ 0.5, 0.5�� ����
    weight = np.array([0.5,0.5])
    
    # 1-1. AND gate�� �����ϴ� bias�� �����մϴ�.
    bias = -0.75
    
    # 1-2. ����ġ, �Է°�, ������ �̿��Ͽ� ���� ��ȣ�� ������ ���մϴ�.
    y = np.sum(x * weight) + bias
    
    # Step Function �Լ��� ȣ���Ͽ� AND gate�� ��°��� ��ȯ�մϴ�.
    return Step_Function(y)

# 2. OR gate �Լ��� �����ϼ���.
def OR_gate(x1, x2):
    x = np.array([x1, x2])
    
    # x1�� x2�� ���� ������ ����ġ 0.5, 0.5�� ����
    weight = np.array([0.5,0.5])
    
    # 2-1. OR gate�� �����ϴ� bias�� �����մϴ�.
    bias = 0
    
    # 2-2. ����ġ, �Է°�, ������ �̿��Ͽ� ���� ��ȣ�� ������ ���մϴ�.
    y = np.sum(x * weight) + bias
    
    #Step Function �Լ��� ȣ���Ͽ� AND gate�� ��°��� ��ȯ�մϴ�.
    return Step_Function(y)

# 3. Step Function ����
def Step_Function(y):
    if y > 0:
        return 1
    else:
        return 0   
    
def main():
    
    # AND Gate�� OR Gate�� �־��� Input �Դϴ�.
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # AND Gate�� �����ϴ��� ����Ͽ� Ȯ���մϴ�.
    print('AND Gate ���')
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ',AND_gate(x1, x2))
        
    # OR Gate�� �����ϴ��� ����Ͽ� Ȯ���մϴ�.
    print('\nOR Gate ���')
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ',OR_gate(x1, x2))

if __name__ == "__main__":
    main()