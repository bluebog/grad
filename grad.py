import numpy as np
lr=0.01
w=np.array([1.0, 2.0])
dw=np.array([0.1, 0.5])
def sgd(w, dw, lr):
    return w-dw*lr
gamma = 0.9
v=np.zeros_like(w)
def momentum(w, dw, lr, gamma, v):
    v=gamma*v-lr*dw
    w=w+v
    return w,v
