import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics
import numpy as np

from simplified_model import SimplifiedModel

# You can construct any distribution strategy you like and use
# it with the SimplifiedModel class.  This includes TPU
# and ParameterServerStrategies.
try:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print("Using OneDeviceStrategy")
except:  # pylint: disable=bare-except
    strategy = tf.distribute.get_strategy()
    print("Using", strategy)

x, y = np.zeros((1000, 10)), np.ones((1000, 1))
ds = tf.data.Dataset.from_tensor_slices((x, y))
# Our model expects the dataset to be batched
ds = ds.batch(10)

ds_test = tf.data.Dataset.from_tensor_slices((x))
ds_test = ds_test.batch(10)

# Make sure to create all `tf.Variable`s under the `scope`.
with strategy.scope():
    # Unfortunately due to an internal bug with keras.Optimizers,
    # we currently need to run all of this code under the same
    # strategy.scope() call.  This will be fixed in a future
    # release.
    model = SimplifiedModel()
    model.compile(
        "sgd",
        loss=losses.MeanSquaredError(reduction=losses.Reduction.SUM),
        metrics=[metrics.MeanAbsolutePercentageError()],
        steps_per_execution=5,
    )

    metrics = model.evaluate(ds)
    print("Metrics Before Fit:", metrics)

    model.fit(ds, epochs=10, verbose=2)

    # We can go ahead and predict some values
    y_pred = model.predict(ds_test)
    print("Predictions Shape:", y_pred.shape)

    # We can also evaluate our set of metrics
    metrics = model.evaluate(ds)
    print("Metrics After:", metrics)
