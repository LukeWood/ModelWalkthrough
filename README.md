# Keras Model Implementation Walkthrough
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukewood/ModelWalkthrough/blob/master/notebooks/ModelWalkthrough.ipynb)

The Keras Model class is one of the centerpieces of the framework.
It encapsulates metric tracking, callbacks, distribution, training loops, various input types, and a wide variety of other training related behavior. 
This has led to the Model class containing a large volume of code that can be intimidating to sift through.

This guide walks through a simplified model implementation in order to help you understand what model does under the hood.
After following along with this guide you will understand how the keras model class achieves the behavior listed above.

The SimplifiedModel can also serve as a starting point for those looking to implement custom models or training loops.
A forkable template using the SimplifiedModel class be found on my github at [https://github.com/lukewood/ModelWalkthrough](https://github.com/lukewood/ModelWalkthrough).

For the sake of brevity, the class written in this guide subclasses `keras.layers.Layer` to leverage some helper functions, such as Keras' `__call__` implementation..  

## Core Implementation
Let's create a basic implementation that supports `compile()`, `fit()`, `predict()`, and `eval()` before we introduce distribution strategy, metric tracking, callbacks, and other features.

Note that the `SimplifiedModel` class operates as a `keras.Model` subclass, overriding the `call()` method to produce predictions.


```python
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optimizers

class SimplifiedModel(keras.layers.Layer):
  """SimplifiedModel is a stripped down barebones version of keras.Model."""

  def __init__(self, *args, **kwargs):
    super(SimplifiedModel, self).__init__(*args, **kwargs)
    self.dense = keras.layers.Dense(1)
    self.distribute_strategy = tf.distribute.get_strategy()
  
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None):
    with self.distribute_strategy.scope():
      self.optimizer = optimizers.get(optimizer)
      self.loss = loss
      self.metrics_list = metrics if isinstance(metrics, list) else [metrics]

  def call(self, inputs):
    return self.dense(inputs)

  def predict_step(self, x):
    return self(x, training=False)

  def train_step(self, data):
    x, y = data
    # Run forward pass.
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.loss(y, y_pred)
      for extra_loss in self.losses:
        loss += scale_loss_for_distribution(extra_loss)

    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

    return_metrics = {'loss': loss}
    for metric in self.metrics_list:
      metric.update_state(y, y_pred, None)
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics

  def fit(self, dataset: tf.data.Dataset, epochs=1, verbose=1):
    """This simplified version of fit only accepts a TensorFlow dataset.

    Args:
      dataset: tf.data.Dataset, must yield a tuple of inputs and a one hot
        encoded vector containing labels
      epochs: number of passes to perform over the verbosity
      verbose: verbosity of logging during fit
    """
    for epoch in range(epochs):
      for batch in dataset:
        metrics = self.train_step(batch)
        metric_str = ', '.join(
            [f'{metric_name}: {val}' for metric_name, val in metrics.items()])
        # Minimal progress logging implementation
        print(f'\repoch: ({epoch+1}/{epochs}), {metric_str}', end='')
    print()

  def test_step(self, x, y):
    y_pred = self(x, training=False)
    loss = self.loss(y, y_pred)
    for extra_loss in self.losses:
      loss += scale_loss_for_distribution(extra_loss)

    return_metrics = {
        'loss': loss,
    }

    for metric in self.metrics_list:
      metric.update_state(y, y_pred, None)
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics

  def evaluate(self, dataset):
    self.reset_metrics()
    metrics_aggregate = []
    for xs, ys in dataset:
      self.reset_metrics()
      metrics_aggregate.append(self.test_step(xs, ys))

    if not metrics_aggregate:
      raise ValueError('dataset must contain at least one batch of samples.  '
                       f'Received: {dataset}')

    result = {}
    for k in metrics_aggregate[0]:
      result[k] = 0.

    for metric_iter in metrics_aggregate:
      for k, v in metric_iter.items():
        result[k] += v / len(metrics_aggregate)
    return result

  def predict(self, dataset):
    result = []
    for xs in dataset:
      result.append(self(xs, training=False))
    return tf.concat(result, axis=0)

  def reset_metrics(self):
    for metric in self.metrics_list:
      metric.reset_state()
```

When you first construct a model, it exists in an uncompiled state.
In this state the optimizer, compiled metrics, and loss have not yet been created.
The `compile()` method creates the optimizer, takes a loss function, and creates a reference to a list of metrics.  These are all later used in training.

`train_step`, `predict_step`, and `eval_step` all contain the logic to perform a single step of their corresponding methods: `fit()`, `predict()`, and `evaluate()` respectively.
Note that while `predict_step()` simply invokes call, `train_step()` and `eval_step()` track loss and metrics.

The simplified model class expects a `tf.data.Dataset` as an input to `fit()`, `predict()`, and `evaluate()`.
The model offloads the batching behavior to the dataset.
To use the model, you'd do something like this:


```python
import numpy as np
import tensorflow.keras.losses as losses

model = SimplifiedModel()

x, y = np.zeros((1000, 10)), np.ones((1000, 1))
ds = tf.data.Dataset.from_tensor_slices((x, y))
# Our model expects the dataset to be batched
ds = ds.batch(10)

ds_test = tf.data.Dataset.from_tensor_slices((x))
ds_test = ds_test.batch(10)

model.compile('sgd', 
              loss=losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM), 
              metrics=[tf.keras.metrics.MeanAbsolutePercentageError()]
)
model.build(input_shape=(None, 10))
metrics = model.evaluate(ds)
print('Metrics Before Fit:', metrics)

model.fit(ds, epochs=10, verbose=2)

metrics = model.evaluate(ds)
print('Metrics After Fit:', metrics)
```

    Metrics Before Fit: {'loss': <tf.Tensor: shape=(), dtype=float32, numpy=10.000002>, 'mean_absolute_percentage_error': <tf.Tensor: shape=(), dtype=float32, numpy=100.0>}
    epoch: (10/10), loss: 1.4210854715202004e-13, mean_absolute_percentage_error: 0.5994006395339966
    Metrics After Fit: {'loss': <tf.Tensor: shape=(), dtype=float32, numpy=1.4210857e-13>, 'mean_absolute_percentage_error': <tf.Tensor: shape=(), dtype=float32, numpy=1.1920929e-05>}


This model class implements our expected behavior, but it's missing some critical logic that `keras.Model` implements.

Perhaps most notably, this model does not support distribution.

## Batched Execution, Compiled `train_step()`

Currently we are executing our `train_step()` calls one at a time, the `train_step()` function is not a compiled `tf.function`, and the model does not work the TensorFlow distribution strategies.  In this section we will implement all of these performance enhancements and net massive performance gains.

First, we begin by wrapping `train_step()` in a compiled function:

```python
class SimplifiedModel(keras.Model):
  # ... 
  def make_train_function(self):
    if self.train_function:
      return self.train_function

    def step_function(model, iterator):

      def run_step(data):
        outputs = model.train_step(data)
        model._train_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      return model.distribute_strategy.unwrap(outputs)[0]

    def train_function(iterator):
      """Runs a training execution with multiple steps."""
      # Autograph cannot infer the return type of undeclared non-Tensor
      # variables from inside loops. The limitations documentation explains this
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md
      names = [m.name for m in self.metrics_list] + ['loss']
      outputs = dict.fromkeys(names, 0.)
      for _ in tf.range(self.steps_per_execution):
        outputs = step_function(self, iterator)
      return outputs

    train_function = tf.function(train_function, experimental_relax_shapes=True)

    # A separate function is needed to prevent self-referential
    # infinitely-recursive closures
    cluster_train_function = None
    if self._cluster_coordinator:
      # pylint: disable=g-long-lambda
      cluster_train_function = lambda it: self._cluster_coordinator.schedule(
          train_function, args=(it,))

    self.train_function = cluster_train_function or train_function
    return self.train_function
```

This function also runs the `train_step()` function using the model's provided distribution strategy.

Next, we need to update our `fit()` method to utilize this new train function:

```python
  def fit(self, dataset: tf.data.Dataset, epochs=1, verbose=1):
    """This simplified version of fit only accepts a TensorFlow dataset.

    Args:
      dataset: tf.data.Dataset, must yield a tuple of inputs and a one hot
        encoded vector containing labels
      epochs: number of passes to perform over the verbosity
      verbose: verbosity of logging during fit
    """
    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
          self.distribute_strategy)
      dataset = self._cluster_coordinator.create_per_worker_dataset(dataset)

    for epoch in range(epochs):
        iterator = iter(dataset)
        for step in range(0, steps_per_epoch, self.steps_per_execution):
          try:
            # returns {'loss': loss, 'metric1': val1, ...}
            metrics = self.train_function(iterator)
            metric_str = ', '.join(
              [f'{metric_name}: {val}' for metric_name, val in metrics.items()])
            print(f'\repoch: ({epoch+1}/{epochs}), {metric_str}', end='')
          except tf.errors.OutOfRangeError:
            break
        print()
```



Next, we will implement computation batching.  By batching computation, we reduce the number of context transfers between the computation host and the python side callbacks.  In the Keras model class this is done by the `steps_per_execution` parameter passed to the `compile()` method.

First, we need to update `compile()` to include this new parameter:

```python
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              steps_per_execution=1):
    # We need to compile the loss and metrics within the strategy scope
    with self.distribute_strategy.scope():
      self.steps_per_execution = steps_per_execution
      ...
```



## Keras Callbacks
Keras callbacks are objects that perform actions at various stages of training.  There is a large library of existing callbacks to handle things like:
- Write TensorBoard logs after every batch of training to monitor your metrics
- Periodically save your model to disk
- Do early stopping
- Get a view on internal states and statistics of a model during training
- ...and more

You can read more about callbacks here: https://keras.io/api/callbacks/

Let's integrate Keras callbacks into our SimplifiedModel class.

In order to do so, we will need to add a callbacks parameter to our `fit()` method.
Additionally, we will wrap the `callbacks` into a Keras `CallbackList`:
```python

  def fit(
      self,
      dataset,
      epochs=1,
      verbose=1,
      steps_per_epoch=sys.maxsize,  # default to max to iterate entire dataset
      callbacks=None):
    """This simplified version of fit only accepts a TensorFlow dataset.

    Args:
      dataset: tf.data.Dataset, must yield a tuple of inputs and a one hot
        encoded vector containing labels
      epochs: number of passes to perform over the verbosity
      verbose: verbosity of logging during fit
      steps_per_epoch: number of steps that counts as an epoch, useful with
        endless datasets.  When using a finite dataset, leave as sys.maxsize.
      callbacks: list of Keras callbacks
    """
    callbacks = callbacks_module.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=verbose != 0,
        model=self,
        verbose=verbose,
        epochs=epochs)

    dataset = self.distribute_strategy.experimental_distribute_dataset(dataset)

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
          self.distribute_strategy)
      dataset = self._cluster_coordinator.create_per_worker_dataset(dataset)

    self.make_train_function()
    self._train_counter.assign(0)
    callbacks.on_train_begin()
    for epoch in range(epochs):
      iterator = iter(dataset)
      callbacks.on_epoch_begin(epoch)
      for step in range(0, steps_per_epoch, self.steps_per_execution):
        callbacks.on_train_batch_begin(step)
        try:
          # returns {'loss': loss, 'metric1': val1, ...}
          unused_metrics = self.train_function(iterator)
        except tf.errors.OutOfRangeError:
          break
        callbacks.on_train_batch_end(step)
      callbacks.on_epoch_end(epoch, None)
```



We can now pass any Keras callbacks to `fit()` and have it behave as expected.  Additionally, we now get the Keras progress bar when running fit.

## Final Usage, Recap
The final code for the SimplifiedModel class is available below:



```python
"""SimplifiedModel is a barebones Keras model class.

The intended use of this class is for end users to fork this class and replace
`compile()`, `fit()` and `predict()` with their own logic.
"""
import sys

import tensorflow as tf


class SimplifiedModel(tf.keras.layers.Layer):
  """SimplifiedModel is a stripped down barebones version of keras.Model."""
  
  def __init__(self, *args, **kwargs):
    super(SimplifiedModel, self).__init__(*args, **kwargs)
    self.dense = tf.keras.layers.Dense(1)
    self.distribute_strategy = tf.distribute.get_strategy()
    agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
    self._train_counter = tf.Variable(0, dtype='int64', aggregation=agg, trainable=False)
    self._cluster_coordinator = None
    if self.distribute_strategy._should_use_with_coordinator:
      self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
          self.distribute_strategy)

  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              steps_per_execution=1):
    # We need to compile the loss and metrics within the strategy scope
    with self.distribute_strategy.scope():
      self.optimizer = optimizers.get(optimizer)
      self.loss = loss
      self.metrics_list = metrics if isinstance(metrics, list) else [metrics]
      self.steps_per_execution = steps_per_execution
      self.train_function = None
      self._is_compiled = True

  def call(self, inputs):
    return self.dense(inputs)

  def predict_step(self, x):
    return self(x, training=False)

  def train_step(self, data):
    x, y = data
    # Run forward pass.
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.loss(y, y_pred)
      for extra_loss in self.losses:
        loss += scale_loss_for_distribution(extra_loss)

    # Run backwards pass.
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    # Collect metrics to return
    return_metrics = {'loss': loss}
    for metric in self.metrics:
      metric.update_state(y, y_pred, None)
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics

  def fit(
      self,
      dataset,
      epochs=1,
      verbose=1,
      steps_per_epoch=sys.maxsize,  # default to max to iterate entire dataset
      callbacks=None):
    """This simplified version of fit only accepts a TensorFlow dataset.

    Args:
      dataset: tf.data.Dataset, must yield a tuple of inputs and a one hot
        encoded vector containing labels
      epochs: number of passes to perform over the verbosity
      verbose: verbosity of logging during fit
      steps_per_epoch: number of steps that counts as an epoch, useful with
        endless datasets.  When using a finite dataset, leave as sys.maxsize.
      callbacks: list of Keras callbacks
    """
    callbacks = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=verbose != 0,
        model=self,
        verbose=verbose,
        epochs=epochs)

    dataset = self.distribute_strategy.experimental_distribute_dataset(dataset)

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
          self.distribute_strategy)
      dataset = self._cluster_coordinator.create_per_worker_dataset(dataset)

    self.make_train_function()
    self._train_counter.assign(0)
    callbacks.on_train_begin()
    for epoch in range(epochs):
      iterator = iter(dataset)
      callbacks.on_epoch_begin(epoch)
      for step in range(0, steps_per_epoch, self.steps_per_execution):
        callbacks.on_train_batch_begin(step)
        try:
          # returns {'loss': loss, 'metric1': val1, ...}
          unused_metrics = self.train_function(iterator)
        except tf.errors.OutOfRangeError:
          break
        callbacks.on_train_batch_end(step)
      callbacks.on_epoch_end(epoch, None)

  def make_train_function(self):
    if self.train_function:
      return self.train_function

    def step_function(model, iterator):

      def run_step(data):
        outputs = model.train_step(data)
        model._train_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      return model.distribute_strategy.unwrap(outputs)[0]

    def train_function(iterator):
      """Runs a training execution with multiple steps."""
      # Autograph cannot infer the return type of undeclared non-Tensor
      # variables from inside loops. The limitations documentation explains this
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md
      names = [m.name for m in self.metrics] + ['loss']
      outputs = dict.fromkeys(names, 0.)
      for _ in tf.range(self.steps_per_execution):
        outputs = step_function(self, iterator)
      return outputs

    train_function = tf.function(train_function, experimental_relax_shapes=True)

    # A separate function is needed to prevent self-referential
    # infinitely-recursive closures
    cluster_train_function = None
    if self._cluster_coordinator:
      # pylint: disable=g-long-lambda
      cluster_train_function = lambda it: self._cluster_coordinator.schedule(
          train_function, args=(it,))

    self.train_function = cluster_train_function or train_function
    return self.train_function

  def test_step(self, x, y):
    y_pred = self(x, training=False)
    loss = self.loss(y, y_pred)
    for extra_loss in self.losses:
      loss += scale_loss_for_distribution(extra_loss)

    return_metrics = {
        'loss': loss,
    }

    for metric in self.metrics:
      metric.update_state(y, y_pred, None)
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics

  def evaluate(self, dataset):
    self.reset_metrics()
    metrics_aggregate = []
    for xs, ys in dataset:
      self.reset_metrics()
      metrics_aggregate.append(self.test_step(xs, ys))

    if not metrics_aggregate:
      raise ValueError('dataset must contain at least one batch of samples.  '
                       f'Received: {dataset}')

    result = {}
    for k in metrics_aggregate[0]:
      result[k] = 0.

    for metric_iter in metrics_aggregate:
      for k, v in metric_iter.items():
        result[k] += v / len(metrics_aggregate)
    return result

  def predict(self, dataset):
    result = []
    for xs in dataset:
      result.append(self(xs, training=False))
    return tf.concat(result, axis=0)

  def reset_metrics(self):
    for metric in self.metrics_list:
      metric.reset_state()
```

Below is an example use of the SimplifiedModel class:


```python
import numpy as np
import tensorflow.keras.losses as losses

try:
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
  print("Using OneDeviceStrategy")
except:
  strategy = tf.distribute.get_strategy()
  print("Using", strategy)

# Make sure to create all `tf.Variable`s under the `scope`.
with strategy.scope():
  model = SimplifiedModel()

  x, y = np.zeros((1000, 10)), np.ones((1000, 1))
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  # Our model expects the dataset to be batched
  ds = ds.batch(10)

  ds_test = tf.data.Dataset.from_tensor_slices((x))
  ds_test = ds_test.batch(10)

  model.compile('sgd',
    loss=losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()],
    steps_per_execution=5
  )
  metrics = model.evaluate(ds)
  print('Metrics Before Fit:', metrics)

  model.fit(ds, epochs=10, verbose=2)

  # we can go ahead and predict some values
  y_pred = model.predict(ds_test)
  print('Predictions Shape:', y_pred.shape)

  metrics = model.evaluate(ds)
  print('Metrics After:', metrics)
```

    Using OneDeviceStrategy
    Metrics Before Fit: {'loss': <tf.Tensor: shape=(), dtype=float32, numpy=10.000002>, 'mean_absolute_percentage_error': <tf.Tensor: shape=(), dtype=float32, numpy=100.0>}
    Epoch 1/10
    101/101 - 1s - 538ms/epoch - 5ms/step
    Epoch 2/10
    101/101 - 0s - 214ms/epoch - 2ms/step
    Epoch 3/10
    101/101 - 0s - 234ms/epoch - 2ms/step
    Epoch 4/10
    101/101 - 0s - 223ms/epoch - 2ms/step
    Epoch 5/10
    101/101 - 0s - 221ms/epoch - 2ms/step
    Epoch 6/10
    101/101 - 0s - 223ms/epoch - 2ms/step
    Epoch 7/10
    101/101 - 0s - 220ms/epoch - 2ms/step
    Epoch 8/10
    101/101 - 0s - 213ms/epoch - 2ms/step
    Epoch 9/10
    101/101 - 0s - 248ms/epoch - 2ms/step
    Epoch 10/10
    101/101 - 0s - 251ms/epoch - 2ms/step
    Predictions Shape: (1000, 1)
    Metrics After: {'loss': <tf.Tensor: shape=(), dtype=float32, numpy=1.4210857e-13>, 'mean_absolute_percentage_error': <tf.Tensor: shape=(), dtype=float32, numpy=1.1920929e-05>}



```python
%%timeit
model.fit(ds, epochs=10, verbose=2)
```

    Epoch 1/10
    101/101 - 0s - 482ms/epoch - 5ms/step
    Epoch 2/10
    101/101 - 0s - 276ms/epoch - 3ms/step
    Epoch 3/10
    101/101 - 0s - 255ms/epoch - 3ms/step
    Epoch 4/10
    101/101 - 0s - 224ms/epoch - 2ms/step
    Epoch 5/10
    101/101 - 0s - 234ms/epoch - 2ms/step
    Epoch 6/10
    101/101 - 0s - 221ms/epoch - 2ms/step
    Epoch 7/10
    101/101 - 0s - 249ms/epoch - 2ms/step
    Epoch 8/10
    101/101 - 0s - 244ms/epoch - 2ms/step
    Epoch 9/10
    101/101 - 0s - 240ms/epoch - 2ms/step
    Epoch 10/10
    101/101 - 0s - 245ms/epoch - 2ms/step
    Epoch 1/10
    101/101 - 0s - 219ms/epoch - 2ms/step
    Epoch 2/10
    101/101 - 0s - 225ms/epoch - 2ms/step
    Epoch 3/10
    101/101 - 0s - 243ms/epoch - 2ms/step
    Epoch 4/10
    101/101 - 0s - 222ms/epoch - 2ms/step
    Epoch 5/10
    101/101 - 0s - 225ms/epoch - 2ms/step
    Epoch 6/10
    101/101 - 0s - 280ms/epoch - 3ms/step
    Epoch 7/10
    101/101 - 0s - 272ms/epoch - 3ms/step
    Epoch 8/10
    101/101 - 0s - 272ms/epoch - 3ms/step
    Epoch 9/10
    101/101 - 0s - 262ms/epoch - 3ms/step
    Epoch 10/10
    101/101 - 0s - 244ms/epoch - 2ms/step
    Epoch 1/10
    101/101 - 0s - 229ms/epoch - 2ms/step
    Epoch 2/10
    101/101 - 0s - 278ms/epoch - 3ms/step
    Epoch 3/10
    101/101 - 0s - 260ms/epoch - 3ms/step
    Epoch 4/10
    101/101 - 0s - 278ms/epoch - 3ms/step
    Epoch 5/10
    101/101 - 0s - 270ms/epoch - 3ms/step
    Epoch 6/10
    101/101 - 0s - 249ms/epoch - 2ms/step
    Epoch 7/10
    101/101 - 0s - 244ms/epoch - 2ms/step
    Epoch 8/10
    101/101 - 0s - 272ms/epoch - 3ms/step
    Epoch 9/10
    101/101 - 0s - 269ms/epoch - 3ms/step
    Epoch 10/10
    101/101 - 0s - 225ms/epoch - 2ms/step
    Epoch 1/10
    101/101 - 0s - 240ms/epoch - 2ms/step
    Epoch 2/10
    101/101 - 0s - 275ms/epoch - 3ms/step
    Epoch 3/10
    101/101 - 0s - 233ms/epoch - 2ms/step
    Epoch 4/10
    101/101 - 0s - 225ms/epoch - 2ms/step
    Epoch 5/10
    101/101 - 0s - 239ms/epoch - 2ms/step
    Epoch 6/10
    101/101 - 0s - 246ms/epoch - 2ms/step
    Epoch 7/10
    101/101 - 0s - 227ms/epoch - 2ms/step
    Epoch 8/10
    101/101 - 0s - 262ms/epoch - 3ms/step
    Epoch 9/10
    101/101 - 0s - 245ms/epoch - 2ms/step
    Epoch 10/10
    101/101 - 0s - 230ms/epoch - 2ms/step
    Epoch 1/10
    101/101 - 0s - 246ms/epoch - 2ms/step
    Epoch 2/10
    101/101 - 0s - 237ms/epoch - 2ms/step
    Epoch 3/10
    101/101 - 0s - 227ms/epoch - 2ms/step
    Epoch 4/10
    101/101 - 0s - 224ms/epoch - 2ms/step
    Epoch 5/10
    101/101 - 0s - 250ms/epoch - 2ms/step
    Epoch 6/10
    101/101 - 0s - 247ms/epoch - 2ms/step
    Epoch 7/10
    101/101 - 0s - 224ms/epoch - 2ms/step
    Epoch 8/10
    101/101 - 0s - 240ms/epoch - 2ms/step
    Epoch 9/10
    101/101 - 0s - 274ms/epoch - 3ms/step
    Epoch 10/10
    101/101 - 0s - 261ms/epoch - 3ms/step
    Epoch 1/10
    101/101 - 0s - 268ms/epoch - 3ms/step
    Epoch 2/10
    101/101 - 0s - 268ms/epoch - 3ms/step
    Epoch 3/10
    101/101 - 0s - 266ms/epoch - 3ms/step
    Epoch 4/10
    101/101 - 0s - 266ms/epoch - 3ms/step
    Epoch 5/10
    101/101 - 0s - 281ms/epoch - 3ms/step
    Epoch 6/10
    101/101 - 0s - 270ms/epoch - 3ms/step
    Epoch 7/10
    101/101 - 0s - 263ms/epoch - 3ms/step
    Epoch 8/10
    101/101 - 0s - 251ms/epoch - 2ms/step
    Epoch 9/10
    101/101 - 0s - 260ms/epoch - 3ms/step
    Epoch 10/10
    101/101 - 0s - 271ms/epoch - 3ms/step
    1 loop, best of 5: 3.02 s per loop


As you can see, `model.fit()` now runs *significantly* faster than it did before implementing our performance enhancements.

# Conclusion
The `keras.Model` class contains a encapsulates set of functionality related to training.  Due to this, the lifecycle of the `keras.Model` class is significantly complex.

The `SimplifiedModel` class we implemented shows how the core functionality of the `keras.Model` works, while still remaining terse and readable.  while the `SimplifiedModel` class is missing a large portion of the true `keras.Model` class's functionality, it is still as useful starting point for implementing custom training loops that work with TensorFlow distribution strategies.

A forkable template using the `SimplifiedModel` class is available at [https://github.com/lukewood/ModelWalkthrough](https://github.com/lukewood/ModelWalkthrough).


