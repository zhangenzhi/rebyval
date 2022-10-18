import os
import numpy as np
import pandas as pd
import tensorflow as tf

from ..tools.utils import check_mkdir


def normalize(x, y):
    dx = []
    dy = []
    for a, b in zip(x, y):
        dx += a/(tf.norm(a)+1e-7)
        dy += b/(tf.norm(b)+1e-7)
    return dx, dy
        
def save_directions(x, y,  save_to=None):
    heads = [x, y]

def get_random_directions(weights, save_to=None):
    x = []
    y = []
    
    for w in weights:
        x += tf.random.uniform(shape=w.shape)
        y += tf.random.uniform(shape=w.shape)
    
    #normalize
    x, y = normalize(x, y)
    
    if save_to ==None:
        save_to = './log/directions.csv'
        check_mkdir(save_to)
    else:
        save_to = os.path.join(save_to, 'directions.csv')
        check_mkdir(save_to)
    save_directions(x=x, y=y, save_to=save_to)
    
    return x, y

def evaluate(model, samples, loss_fn):
    pred = model(samples['inputs'])
    loss = loss_fn(pred, samples['labels'])
    return loss

def weights_update(weights, dx, dy):
    new_weights = []
    for w,x,y in zip(weights,dx,dy):
        new_weights += w+x+y
    return new_weights

def save_lossland(lossland, save_to=None):
    np.savetxt(save_to, lossland, delimiter=",")

def visualization(model, samples, loss_fn, step_size=1e-2, save_to=None):
    weights = model.trainable_variables
    x, y = get_random_directions(weights=weights, save_to=save_to)
    
    N = 1/step_size * 2
    lossland = np.zeros(shape=(2*N, 2*N))
    for a in range(-N,N):
        for b in range(-N, N):
            weights_update(weights, a*x, b*y)
            lossland[a][b] = evaluate(model=model, samples=samples, loss_fn=loss_fn)
    
    if save_to ==None:
        save_to = './log/losslands.csv'
        check_mkdir(save_to)
    else:
        check_mkdir(save_to)
        
    save_lossland(x=x, y=y, save_to=save_to)
    