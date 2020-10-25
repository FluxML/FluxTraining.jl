# Features

This page gives a run-down of many features *FluxTraining.jl* brings to the table.

Most features are implemented as callbacks and using them is as simple as passing the callback when constructing the [`Learner`](#):

```julia
cb = CoolFeature🕶️Callback()
learner = Learner(model, data, opt, lossfn, cb)
```

## Basics

[`Learner`](#) holds all training state. Use `fit!(learner, n)` to train it for `n` epochs.

## Metrics

By default, `Learner` will track only the loss function. You can track other metric with the [`Metrics`](#) callback. See also [`Metric`](#), [`AbstractMetric`](#).

## Hyperparameter scheduling

The [`Scheduler`](#) callback takes care of hyperparameter scheduling. See the [Hyperparameter scheduling tutorial] and also [`Scheduler`](#), [`Schedule`](#), [`HyperParameter`](#).

## Logging

For logging, use the [`Logger`](#) callback. It can have multiple backends but right now only [`TensorBoardBackend`](#) is part of *FluxTraining.jl*. See also [`LoggerBackend`](#), [`log_to`](#), and [`Loggables.Loggable`](#).

## Checkpointing

Use the [`Checkpointer`](#) callback to create model checkpoints after every epoch.