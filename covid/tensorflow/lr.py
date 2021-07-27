from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

class LRFinder(Callback):
    def __init__(self, min_lr=1e-6, max_lr=0.1, mom=0.9, stop_multiplier=None, reload_weights=True, batches_lr_update=5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20*self.mom/3 + 10 # 4 if mom=0.9
        else:
            self.stop_multiplier = stop_multiplier
        
    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p['epochs']*p['samples']//p['batch_size']
        except:
            n_iterations = p['steps']*p['epochs']
            
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, num=n_iterations//self.batches_lr_update+1)
        self.losses=[]
        self.iteration=0
        self.best_loss=0
        if self.reload_weights:
            self.model.save_weights('tmp.hdf5')
        
    
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        
        if self.iteration!=0: # Make loss smoother using momentum
            loss = self.losses[-1]*self.mom+loss*(1-self.mom)
        
        if self.iteration==0 or loss < self.best_loss: 
                self.best_loss = loss
                
        if self.iteration%self.batches_lr_update==0: # Evaluate each lr over 5 epochs
            
            if self.reload_weights:
                self.model.load_weights('tmp.hdf5')
            lr = self.learning_rates[self.iteration//self.batches_lr_update]            
            K.set_value(self.model.optimizer.lr, lr)

            self.losses.append(loss)            

        if loss > self.best_loss*self.stop_multiplier: # Stop criteria
            self.model.stop_training = True
                
        self.iteration += 1
    
    def on_train_end(self, logs=None):
        if self.reload_weights:
                self.model.load_weights('tmp.hdf5')
                
        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()
        
        
        
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K


class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """
    def __init__(self, warmup_batches, init_lr, verbose=2):
        """Constructor for warmup learning rate scheduler
        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.
        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning rate to %s.' % (self.batch_count + 1, lr))
                
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=1):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            if batch < 10 or batch // 64 == 0: 
                print('\nBatch %05d: setting lr to %s.' % (self.global_step + 1, lr))


from tensorflow.python.keras.optimizer_v2 import *
import abc
import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
@keras_export("keras.optimizers.schedules.LearningRateSchedule")
class LearningRateSchedule(object):
    """The learning rate schedule base class.
    You can use a learning rate schedule to modulate how the learning rate
    of your optimizer changes over time.
    Several built-in learning rate schedules are available, such as
    `tf.keras.optimizers.schedules.ExponentialDecay` or
    `tf.keras.optimizers.schedules.PiecewiseConstantDecay`:
    ```python
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=1e-2,
          decay_steps=10000,
          decay_rate=0.9)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    ```
    A `LearningRateSchedule` instance can be passed in as the `learning_rate`
    argument of any optimizer.
    To implement your own schedule object, you should implement the `__call__`
    method, which takes a `step` argument (scalar integer tensor, the
    current training step count).
    Like for any other Keras object, you can also optionally
    make your object serializable by implementing the `get_config`
    and `from_config` methods.
    Example:
    ```python
    class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
      def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
    def __call__(self, step):
       return self.initial_learning_rate / (step + 1)
  optimizer = tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))
  ```
  """

    @abc.abstractmethod
    def __call__(self, step):
        raise NotImplementedError("Learning rate schedule must override __call__")

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError("Learning rate schedule must override get_config")

    @classmethod
    def from_config(cls, config):
        """Instantiates a `LearningRateSchedule` from its config.
        Args:
            config: Output of `get_config()`.
        Returns:
            A `LearningRateSchedule` instance.
        """
        return cls(**config)
@keras_export("keras.optimizers.schedules.CosineDecayRestarts")
class CosineDecayRestarts(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with restarts.
    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.
    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function with
    restarts to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    The learning rate multiplier first decays
    from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
    restart is performed. Each new warm restart runs for `t_mul` times more
    steps and with `m_mul` times smaller initial learning rate.
    Example usage:
    ```python
        first_decay_steps = 1000
        lr_decayed_fn = (
        tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
    """
    def __init__(self, initial_learning_rate, first_decay_steps, train_steps, t_mul=2.0, m_mul=1.0, alpha=0, name=None): 
        super(CosineDecayRestarts, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name
        self.train_steps = train_steps

    def __call__(self, step): 
        # First 5 epochs: Very gradual warmup lr
        if step < self.train_steps: 
            return (step/self.train_steps) * 1e-10
        if step <= 2 * self.train_steps: 
            return 1e-8
        if step <= 3 * self.train_steps: 
            return 1e-6
        if step <= 4 * self.train_steps: 
            return 1e-5
        if step <= 5 * self.train_steps: 
            return 5e-5
        with ops.name_scope_v2(self.name or 'SGDRDecay') as name:
            initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
                self.initial_learning_rate, name="initial_learning_rate")
        dtype = initial_learning_rate.dtype
        first_decay_steps = math_ops.cast(self.first_decay_steps, dtype)
        alpha = math_ops.cast(self.alpha, dtype)
        t_mul = math_ops.cast(self._t_mul, dtype)
        m_mul = math_ops.cast(self._m_mul, dtype)
        global_step_recomp = math_ops.cast(step, dtype)
        completed_fraction = global_step_recomp / first_decay_steps
    
        def compute_step(completed_fraction, geometric=False):
            """Helper for `cond` operation."""
            if geometric:
                i_restart = math_ops.floor(
                    math_ops.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                    math_ops.log(t_mul))

                sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart
            else:
                i_restart = math_ops.floor(completed_fraction)
                completed_fraction -= i_restart
            return i_restart, completed_fraction
        
        i_restart, completed_fraction = control_flow_ops.cond(
            math_ops.equal(t_mul, 1.0),
            lambda: compute_step(completed_fraction, geometric=False),
            lambda: compute_step(completed_fraction, geometric=True))

        m_fac = m_mul**i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
            constant_op.constant(math.pi) * completed_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha

        return math_ops.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name
        }