import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from elice_utils import EliceUtils
import tensorflow as tf

elice_utils = EliceUtils()

def constant_tensors():
    # 5의 값을 가지는 (1,1) shape의 8-bit integer 텐서를 만드세요.
    t1 = tf.constant(5, dtype=tf.int8)
    
    # 모든 원소의 값이 0인 (3,5) shape의 16-bit integer 텐서를 만드세요.
    t2 = tf.zeros((3,5), dtype=tf.int16)
    
    # 모든 원소의 값이 1인 (4,3) shape의 8-bit integer 텐서를 만드세요.
    t3 = tf.ones((4,3), dtype=tf.int8)
    
    return t1, t2, t3
    
def sequence_tensors():
    # 1.5에서 10.5까지 증가하는 3개의 텐서를 만드세요.
    seq_t1 = tf.linspace(1.5,10.5,3)
    
    # 1에서 10까지 2씩 증가하는 텐서를 만드세요.
    seq_t2 = tf.range(1,10,2)
    
    return seq_t1,seq_t2

def random_tensors():
    # 난수를 생성하기 위한 seed 값입니다.
    # 정확한 채점을 위해 값을 변경하지 마세요!
    seed=3921
    tf.random.set_seed(seed)
    
    # 평균이 0이고 표준편차가 1인  정규 분포를 가진 (7,4) shape의 32-bit float 난수 텐서를 만드세요.
    # 정확한 채점을 위하여 미리 설정된 seed 값을 사용해주세요.
    rand_t1 = tf.random.normal((7,4),mean =0,stddev=1,dtype=tf.float32, seed=seed)
    
    # 최소값이 0이고 최대값이 3인 균등 분포를 가진 (5,4,3) shape의 32-bit float 난수 텐서를 만드세요.
    # 정확한 채점을 위하여 미리 설정된 seed 값을 사용해주세요.
    rand_t2 = tf.random.uniform((5,4,3),minval = 0,maxval = 3,dtype=tf.float32, seed=seed)
    
    return rand_t1, rand_t2

def variable_tensor():
    # 값이 100인 변수 텐서를 만드세요.
    var_tensor = tf.Variable(100)
    
    return var_tensor
    
def main():
    # 1. constant_tensors 함수를 완성하세요.
    t1, t2, t3 = constant_tensors()
    
    # 2. sequence_tensors 함수를 완성하세요.
    seq_t1,seq_t2 = sequence_tensors()
    
    # 3. random_tensors 함수를 완성하세요.
    rand_t1, rand_t2 = random_tensors()
    
    # 4. variable_tensor 함수를 완성하세요.
    var_tensor = variable_tensor()
    
    for i in [t1, t2, t3,seq_t1,seq_t2,rand_t1, rand_t2, var_tensor ]:
        print(i.numpy())

if __name__ == "__main__":
    main()