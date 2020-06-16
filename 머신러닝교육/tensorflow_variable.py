import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from elice_utils import EliceUtils
import tensorflow as tf

elice_utils = EliceUtils()

def constant_tensors():
    # 5�� ���� ������ (1,1) shape�� 8-bit integer �ټ��� ���弼��.
    t1 = tf.constant(5, dtype=tf.int8)
    
    # ��� ������ ���� 0�� (3,5) shape�� 16-bit integer �ټ��� ���弼��.
    t2 = tf.zeros((3,5), dtype=tf.int16)
    
    # ��� ������ ���� 1�� (4,3) shape�� 8-bit integer �ټ��� ���弼��.
    t3 = tf.ones((4,3), dtype=tf.int8)
    
    return t1, t2, t3
    
def sequence_tensors():
    # 1.5���� 10.5���� �����ϴ� 3���� �ټ��� ���弼��.
    seq_t1 = tf.linspace(1.5,10.5,3)
    
    # 1���� 10���� 2�� �����ϴ� �ټ��� ���弼��.
    seq_t2 = tf.range(1,10,2)
    
    return seq_t1,seq_t2

def random_tensors():
    # ������ �����ϱ� ���� seed ���Դϴ�.
    # ��Ȯ�� ä���� ���� ���� �������� ������!
    seed=3921
    tf.random.set_seed(seed)
    
    # ����� 0�̰� ǥ�������� 1��  ���� ������ ���� (7,4) shape�� 32-bit float ���� �ټ��� ���弼��.
    # ��Ȯ�� ä���� ���Ͽ� �̸� ������ seed ���� ������ּ���.
    rand_t1 = tf.random.normal((7,4),mean =0,stddev=1,dtype=tf.float32, seed=seed)
    
    # �ּҰ��� 0�̰� �ִ밪�� 3�� �յ� ������ ���� (5,4,3) shape�� 32-bit float ���� �ټ��� ���弼��.
    # ��Ȯ�� ä���� ���Ͽ� �̸� ������ seed ���� ������ּ���.
    rand_t2 = tf.random.uniform((5,4,3),minval = 0,maxval = 3,dtype=tf.float32, seed=seed)
    
    return rand_t1, rand_t2

def variable_tensor():
    # ���� 100�� ���� �ټ��� ���弼��.
    var_tensor = tf.Variable(100)
    
    return var_tensor
    
def main():
    # 1. constant_tensors �Լ��� �ϼ��ϼ���.
    t1, t2, t3 = constant_tensors()
    
    # 2. sequence_tensors �Լ��� �ϼ��ϼ���.
    seq_t1,seq_t2 = sequence_tensors()
    
    # 3. random_tensors �Լ��� �ϼ��ϼ���.
    rand_t1, rand_t2 = random_tensors()
    
    # 4. variable_tensor �Լ��� �ϼ��ϼ���.
    var_tensor = variable_tensor()
    
    for i in [t1, t2, t3,seq_t1,seq_t2,rand_t1, rand_t2, var_tensor ]:
        print(i.numpy())

if __name__ == "__main__":
    main()