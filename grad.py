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
h=np.zeros_like(w)
eps=1e-8
def adagrad(h, w, dw, lr):
    h+=dw*dw
    w=w-lr*(1/(np.sqrt(h)+eps))*dw
    return w,h
