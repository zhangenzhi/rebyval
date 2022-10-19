import os
import numpy as np
import tensorflow as tf

from ..tools.utils import check_mkdir


def normalize(x, y):
    dx = []
    dy = []
    for a, b in zip(x, y):
        dx += [a/(tf.norm(a, axis=0)+1e-7)]
        dy += [b/(tf.norm(b, axis=0)+1e-7)]
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

def get_regular_directions(model,save_to=None):
    pass

def get_sample_directions(model, samples, lt, save_to=None):
    # train-test directions
    dx = []
    dy = []
    with tf.GradientTape() as g:
        tr_pred = model(samples["train"]["inputs"])
        train_loss = lt(samples["train"]["labels"], tr_pred)
    train_grad = g.gradient(train_loss, model.trainable_variables)
    
    with tf.GradientTape() as g:
        te_pred = model(samples["test"]["inputs"])
        test_loss = lt(samples["test"]["labels"], te_pred)
    test_grad = g.gradient(test_loss, model.trainable_variables)
    
    for w, g in zip(train_grad, test_grad):
        dx += [w/(tf.norm(w, axis=0)+1e-7)]
        dy += [g/(tf.norm(g, axis=0)+1e-7)]
        
    return dx, dy

def get_train_norm_directions(model, samples, lt, penalty=0.1, save_to = None):
    # train-l2 norm direction
    dx = []
    dy = []
    with tf.GradientTape() as g:
        pred = model(samples["train"]["inputs"])
        loss = lt(samples["train"]["labels"], pred)
        
    sample_grad = g.gradient(loss, model.trainable_variables)
    for w, g in zip(model.trainable_variables, sample_grad):
        dx += [penalty*2*w]
        dy += [g]
    return dx, dy

def get_random_directions(model, save_to=None):
    weights = model.trainable_variables
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

def evaluate(model, samples, lt, mt, mode="train"):
    
    pred = model(samples[mode]['inputs'])
    loss = lt(samples[mode]["labels"], pred)
    mt.update_state(samples[mode]['labels'], pred)
    mt_v = mt.result() 
    mt.reset_states()
    return loss, mt_v

def get_weights_change(dx, dy):
    diff = []
    for x,y in zip(dx, dy):
        diff += [x+y]
    return diff

def visualization(model, 
                  train_sample, test_sample, 
                  step_size=1e-2, scale=100, save_to=None):
    
    samples = {"train":train_sample, "test":test_sample}
    changer = tf.keras.optimizers.SGD(1.0)
    backer = tf.keras.optimizers.SGD(-1.0)
    lt = tf.keras.losses.CategoricalCrossentropy()
    mt = tf.keras.metrics.CategoricalAccuracy()
    
    # x, y = get_train_norm_directions(model=model, samples=samples, 
    #                                  lt=lt, penalty=1.0,
    #                                  save_to=save_to)
    # x, y = get_random_directions(model=model, save_to=save_to)
    
    x, y = get_sample_directions(model=model, samples=samples, lt=lt, save_to=save_to)
    N = int(scale/2)
    
    tr_lossland = np.zeros(shape=(2*N, 2*N))
    tr_mtland = np.zeros(shape=(2*N, 2*N))
    te_lossland = np.zeros(shape=(2*N, 2*N))
    te_mtland = np.zeros(shape=(2*N, 2*N))
    
    for a in range(-N,N):
        for b in range(-N, N):
            
            diff = get_weights_change([step_size*a*xi for xi in x], [step_size*b*yi for yi in y])
            
            changer.apply_gradients(zip(diff, model.trainable_variables))
            tr_lossland[a+N][b+N], tr_mtland[a+N][b+N] = evaluate(model=model, samples=samples, mode="train", lt=lt, mt=mt)
            te_lossland[a+N][b+N], te_mtland[a+N][b+N] = evaluate(model=model, samples=samples, mode="test", lt=lt, mt=mt)
            backer.apply_gradients(zip(diff, model.trainable_variables))
    
    if save_to == None:
        save_to = './log'
        check_mkdir(save_to)
        save_to = os.path.join(save_to, 'lossland.csv')
        tr_mt_save_to = os.path.join(save_to, 'mtland.csv')
    else:
        check_mkdir(save_to)
        tr_loss_save_to = os.path.join(save_to, 'tr_lossland.csv')
        tr_mt_save_to = os.path.join(save_to, 'tr_mtland.csv')
        te_loss_save_to = os.path.join(save_to, 'te_lossland.csv')
        te_mt_save_to = os.path.join(save_to, 'te_mtland.csv')
        
    np.savetxt(tr_loss_save_to, tr_lossland, delimiter=",")
    np.savetxt(tr_mt_save_to, tr_mtland, delimiter=",")
    np.savetxt(te_loss_save_to, te_lossland, delimiter=",")
    np.savetxt(te_mt_save_to, te_mtland, delimiter=",")
    