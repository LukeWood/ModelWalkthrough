"""SimplifiedModel is a barebones Keras model class.

The intended use of this class is for end users to fork this class and replace
`compile()`, `fit()` and `predict()` with their own logic.
"""
import sys

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.layers as layers


class SimplifiedModel(tf.keras.layers.Layer):
    """SimplifiedModel is a stripped down barebones version of keras.Model."""

    def __init__(self, *args, **kwargs):
        super(SimplifiedModel, self).__init__(*args, **kwargs)
        self.dense = layers.Dense(1)
        self.distribute_strategy = tf.distribute.get_strategy()
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(
            0, dtype="int64", aggregation=agg, trainable=False
        )
        self._cluster_coordinator = None
        if self.distribute_strategy._should_use_with_coordinator:
            self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
                self.distribute_strategy
            )

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        steps_per_execution=1,
    ):
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
        return_metrics = {"loss": loss}
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
        callbacks=None,
    ):
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
            epochs=epochs,
        )

        dataset = self.distribute_strategy.experimental_distribute_dataset(dataset)

        if (
            self.distribute_strategy._should_use_with_coordinator
        ):  # pylint: disable=protected-access
            self._cluster_coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
                self.distribute_strategy
            )
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
            names = [m.name for m in self.metrics] + ["loss"]
            outputs = dict.fromkeys(names, 0.0)
            for _ in tf.range(self.steps_per_execution):
                outputs = step_function(self, iterator)
            return outputs

        train_function = tf.function(train_function, experimental_relax_shapes=True)

        # A separate function is needed to prevent self-referential
        # infinitely-recursive closures
        cluster_train_function = None
        if self._cluster_coordinator:
            # pylint: disable=g-long-lambda
            def cluster_train_function(it):
                return self._cluster_coordinator.schedule(train_function, args=(it,))

        self.train_function = cluster_train_function or train_function
        return self.train_function

    def test_step(self, x, y):
        y_pred = self(x, training=False)
        loss = self.loss(y, y_pred)
        for extra_loss in self.losses:
            loss += scale_loss_for_distribution(extra_loss)

        return_metrics = {
            "loss": loss,
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
            raise ValueError(
                "dataset must contain at least one batch of samples.  "
                f"Received: {dataset}"
            )

        result = {}
        for k in metrics_aggregate[0]:
            result[k] = 0.0

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


def scale_loss_for_distribution(loss_value):
    """Scales and returns the given loss value by the number of replicas."""
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if num_replicas > 1:
        loss_value *= 1.0 / num_replicas
    return loss_value
