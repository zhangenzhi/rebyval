import tensorflow as tf


# Init learning rate by baseline SGD, ex, 0.00001
# After warmup_step, leaning rate *= batch_size
class LinearScalingWithDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, linear_scaling, base_learning_rate=0.01, warmup_steps=4000, gradual_steps=4000,
                 decay_steps=100000):
        super(LinearScalingWithDecaySchedule, self).__init__()

        self.linear_scaling = linear_scaling
        self.linear_scaling = tf.cast(self.linear_scaling, tf.float32)

        self.base_learning_rate = base_learning_rate
        self.base_learning_rate = tf.cast(self.base_learning_rate, tf.float32)
        self.decay_lr = self.base_learning_rate

        self.warmup_steps = warmup_steps
        self.gradual_steps = gradual_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        # lr decay

        # decay_arg = tf.math.sign(self.decay_steps - step)
        # self.decay_steps = tf.math.maximum(self.decay_steps, self.decay_steps * (1 - decay_arg))
        decay_arg = tf.math.floordiv(step, self.decay_steps)
        decay_factor = tf.math.pow(0.1, decay_arg)
        # self.decay_lr = self.decay_lr * decay_factor

        ## constant linear scaling

        # arg1 = tf.math.sign(step - self.warmup_steps)
        # arg2 = self.linear_scaling * arg1
        #
        # return self.base_learning_rate * tf.math.maximum(arg1, arg2) * arg1

        # gradual linear scaling
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.sign(step - self.warmup_steps)

        gradual_factor = arg1 + (self.linear_scaling - 1) * tf.math.minimum(
            (step - self.warmup_steps) / self.gradual_steps, 1)
        arg2 = gradual_factor

        linear_scaling = tf.math.maximum(arg1, arg2) * arg1

        return self.base_learning_rate * decay_factor * linear_scaling
