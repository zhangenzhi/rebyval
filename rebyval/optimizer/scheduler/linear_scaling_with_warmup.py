import tensorflow as tf


# Init learning rate by baseline SGD, ex, 0.00001
# After warmup_step, leaning rate *= batch_size
class LinearScalingWithWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, linear_scaling, base_learning_rate=0.01, warmup_steps=4000, gradual_steps=4000):
        super(LinearScalingWithWarmupSchedule, self).__init__()

        self.linear_scaling = linear_scaling
        self.linear_scaling = tf.cast(self.linear_scaling, tf.float32)

        self.base_learning_rate = base_learning_rate
        self.base_learning_rate = tf.cast(self.base_learning_rate, tf.float32)

        self.warmup_steps = warmup_steps
        self.gradual_steps = gradual_steps
        
        self.current_lr = 0.0

    def __call__(self, step):

        # gradual linear scaling

        step = tf.cast(step, tf.float32)
        arg1 = tf.math.sign(step - self.warmup_steps)

        gradual_factor = arg1 + (self.linear_scaling - 1) * tf.math.minimum((step - self.warmup_steps)/self.gradual_steps, 1)
        arg2 = gradual_factor

        self.current_lr = self.base_learning_rate * tf.math.maximum(arg1, arg2) * arg1
        return self.current_lr