import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten


class dnn(Model):
    def __init__(self, hidden_dims=[128, 64, 1]):
        super(dnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.dnn_layer = self.build_deep_layers()

    def build_deep_layers(self):
        deep_layers = []

        for dim in self.hidden_dims:
            dense_layer = Dense(
                dim,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.keras.initializers.GlorotUniform())
            deep_layers.append(dense_layer)

        return deep_layers

    def call(self, x):
        next_input = Flatten()(x)
        # next_input = next_input * next_input * 5
        for deep_layer in self.dnn_layer:
            next_input = deep_layer(next_input)
        return next_input


class surrogate_dnn(Model):
    def __init__(self, hidden_dims=[128, 64, 1]):
        super(surrogate_dnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.dnn_layer = self.build_deep_layers()

    def build_deep_layers(self):
        deep_layers = []

        for dim in self.hidden_dims:
            dense_layer = Dense(
                dim,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.keras.initializers.GlorotUniform())
            deep_layers.append(dense_layer)

        return deep_layers

    def call(self, x):
        flat_vars = []
        for tensor in x:
            flat_vars.append(tf.reshape(tensor, shape=(-1)))
        flat_vars = tf.concat(flat_vars, axis=0)
        flat_vars = tf.reshape(flat_vars, shape=(1, -1))
        next_input = Flatten()(flat_vars)
        for deep_layer in self.dnn_layer:
            next_input = deep_layer(next_input)
        return next_input


if __name__ == '__main__':
    x = tf.ones(shape=(8, 2))
    y = tf.zeros(shape=(8, 1))

    train_steps=30
    target_model = dnn()
    surrogate_model = surrogate_dnn()
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for i in range(train_steps):
        with tf.GradientTape(persistent=True) as tape:

            predict = target_model(x, training=True)

            s_loss = surrogate_model(target_model.trainable_variables, training=False)
            m_loss = loss_fn(predict, y)
            loss = m_loss + s_loss
            print("train_loss + surrogate_loss:",loss.numpy())

        target_model_grad = tape.gradient(m_loss, target_model.trainable_variables)
        surrogate_model_grad = tape.gradient(s_loss, target_model.trainable_variables)
        consider_grad = []
        for i in range(len(target_model_grad)):
            consider_grad.append(surrogate_model_grad[i] + target_model_grad[i])
        gradients = tape.gradient(loss, target_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, target_model.trainable_variables))
