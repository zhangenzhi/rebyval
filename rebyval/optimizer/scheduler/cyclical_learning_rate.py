import tensorflow as tf


# Init learning rate by baseline SGD, ex, 0.00001
# After warmup_step, leaning rate *= batch_size
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 initial_learning_rate=0.001,
                 maximal_learning_rate=0.1,
                 step_size=None,
                 scale_fn=None,
                 scale_mode=None,
                 name: str = "CyclicalLearningRate"):
        super(CyclicalLearningRate, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.step_size = step_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicalLearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            step_size = tf.cast(self.step_size, dtype)
            step_as_dtype = tf.cast(step, dtype)
            cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
            x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)

            mode_step = cycle if self.scale_mode == "cycle" else step

            return initial_learning_rate + (
                    maximal_learning_rate - initial_learning_rate
            ) * tf.maximum(tf.cast(0, dtype), (1 - x) * self.scale_fn(mode_step))

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "scale_fn": self.scale_fn,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }


class TriangularCyclicalLearningRate(CyclicalLearningRate):

    def __init__(self,
                 initial_learning_rate=0.001,
                 maximal_learning_rate=0.1,
                 step_size=None,
                 scale_mode=None,
                 name: str = "TriangularCyclicalLearningRate"
                 ):
        super(TriangularCyclicalLearningRate, self).__init__(initial_learning_rate=initial_learning_rate,
                                                             maximal_learning_rate=maximal_learning_rate,
                                                             step_size=step_size,
                                                             scale_fn=lambda x: 1.0,
                                                             scale_mode=scale_mode,
                                                             name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }
