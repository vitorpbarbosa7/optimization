from numpy import array as ar
import numpy as np

wc = 0.07142857
w = ar([wc,wc])
b = 0

y_pos = +1
y_neg = -1

examples_pos = ar([[7,7],[11.3,10.2]])
for example in examples_pos:
    print(y_pos*np.dot(w,example) + b)

examples_neg = ar([-7,-7])
print(y_neg*np.dot(w,examples_neg) + b)



