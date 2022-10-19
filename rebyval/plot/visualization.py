import os
import numpy as np
import tensorflow as tf

from ..tools.utils import check_mkdir


def normalize(x, y):
    dx = []
    dy = []
    for a, b in zip(x, y):
        dx += [a/(tf.norm(a)+1e-7)]
        dy += [b/(tf.norm(b)+1e-7)]
    return dx, dy

def unitwise_normalize(x, y, weights):
    dx = []
    dy = []
    for w, a, b in zip(weights, x, y):
        dx += [a/(tf.norm(a, axis=0)+1e-7)*tf.norm(w, axis=0)]
        dy += [b/(tf.norm(b, axis=0)+1e-7)*tf.norm(w, axis=0)]
    return dx, dy
        
def save_directions(x, y,  save_to=None):
    heads = [x, y]

def get_random_directions(weights,save_to=None):
    x = []
    y = []
    
    # for w in weights:
    #     x += [tf.random.uniform(shape=w.shape)]
    #     y += [tf.random.uniform(shape=w.shape)]
    
    for w in weights:
        x += [tf.random.normal(shape=w.shape)]
        y += [tf.random.normal(shape=w.shape)]
    
    #normalize
    # x, y = normalize(x, y)
    x, y = unitwise_normalize(x, y, weights)
    
    if save_to ==None:
        save_to = './log/directions.csv'
        check_mkdir(save_to)
    else:
        save_to = os.path.join(save_to, 'directions.csv')
        check_mkdir(save_to)
    save_directions(x=x, y=y, save_to=save_to)
    
    return x, y

def evaluate(model, samples, loss_fn, mt):
    
    pred = model(samples['inputs'])
    loss = loss_fn(pred, samples['labels'])
    mt.update_state(pred, samples['labels'])
    mt_v = mt.result() 
    mt.reset_states()
    return loss, mt_v

def get_weights_change(dx, dy):
    diff = []
    for x,y in zip(dx, dy):
        diff += [x+y]
    return diff

def visualization(model, samples, loss_fn, step_size=1e-3, scale=100, save_to=None):
    
    weights = model.trainable_variables
    changer = tf.keras.optimizers.SGD(1.0)
    backer = tf.keras.optimizers.SGD(-1.0)
    mt = tf.keras.metrics.CategoricalAccuracy()
    x, y = get_random_directions(weights=weights, save_to=save_to)
    N = int(scale/2)
    
    lossland = np.zeros(shape=(2*N, 2*N))
    mtland = np.zeros(shape=(2*N, 2*N))
    
    for a in range(-N,N):
        for b in range(-N, N):
            diff = get_weights_change([step_size*a*xi for xi in x], [step_size*b*yi for yi in y])
            
            changer.apply_gradients(zip(diff, model.trainable_variables))
            lossland[a][b], mtland[a][b] = evaluate(model=model, samples=samples, loss_fn=loss_fn, mt=mt)
            backer.apply_gradients(zip(diff, model.trainable_variables))
    
    if save_to == None:
        save_to = './log'
        check_mkdir(save_to)
        save_to = os.path.join(save_to, 'lossland.csv')
        mt_save_to = os.path.join(save_to, 'mtland.csv')
    else:
        check_mkdir(save_to)
        loss_save_to = os.path.join(save_to, 'lossland.csv')
        mt_save_to = os.path.join(save_to, 'mtland.csv')
        
    np.savetxt(loss_save_to, lossland, delimiter=",")
    np.savetxt(mt_save_to, mtland, delimiter=",")
    