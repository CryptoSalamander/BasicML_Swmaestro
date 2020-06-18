import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#1. �� �Ǽ��� ���� ������ �Է¹޴� �Լ��Դϴ�. �ڵ带 ���캸����.
def insert():
    x = float(input('���� �Ǵ� �Ǽ��� �Է��ϼ���. x : '))
    y = float(input('���� �Ǵ� �Ǽ��� �Է��ϼ���. y : '))
    cal = input('� ������ �Ұ����� �Է��ϼ���. (+, -, *, /)')
    return x, y, cal

# ��Ģ���� �Լ��� �����غ�����.
def calcul(x,y,cal):
    result = 0
    # ���ϱ�
    if cal == '+':
        result = tf.add(x,y)
    # ����
    if cal == '-':
        result = tf.subtract(x,y)
    # ���ϱ�
    if cal == '*':
        result = tf.multiply(x,y)
    # ������
    if cal == '/':
        result = tf.truediv(x,y)
    return result.numpy()
    
    
def main():
    
    # �� �Ǽ��� ���� ������ �Է¹޴� insert �Լ��� ȣ���մϴ�.
    x, y, cal = insert()

    # calcul �Լ��� ȣ���� �Ǽ� ��Ģ������ �����ϰ� ����� ����غ�����.
    print(calcul(x,y,cal))
    
    
if __name__ == "__main__":
    main()