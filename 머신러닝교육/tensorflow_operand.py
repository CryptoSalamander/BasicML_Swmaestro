import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(10, dtype = tf.int32)
b = tf.constant(3, dtype = tf.int32)
c = tf.constant(-10, dtype = tf.int32)
boolean = tf.constant(True, dtype = tf.bool)

# 1. 단항 연산자를 사용해보세요.
neg = tf.negative(a)
logic = tf.logical_not(boolean)
absolute = tf.abs(c)

# 2. 이항 연산자를 사용해 사칙연산을 수행해보세요.
add = tf.add(a,b)
sub = tf.subtract(a,b)
mul = tf.multiply(a,b)
div = tf.truediv(a,b)

for i in [neg, logic, absolute, add, sub, mul, div]:
    print(i.numpy())