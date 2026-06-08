import numpy as np
import matplotlib.pyplot as plt

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
class adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999 ):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.iter=0
        self.v=None
        self.m=None

    def update(self, params, grad):
        if self.m is None:
            self.v, self.m= {}, {}
            for key, val in params.items():
                self.v[key]=np.zeros_like(val)
                self.m[key]=np.zeros_like(val)
        self.iter +=1
        lr_t=self.lr*np.sqrt(1.0-self.beta1**self.iter)/(1.0-self.beta2**self.iter)
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
    
