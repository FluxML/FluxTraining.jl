# Features

This page gives a run-down of many features *FluxTraining.jl* brings to the table.  Most features are implemented as callbacks and using them is as simple as passing the callback when constructing a [`Learner`](#) and training with [`fit!`](#):

```julia
cb = CoolFeature🕶️Callback()
learner = Learner(model, data, opt, lossfn, cb)
fit!(learner, nepochs)
```

## Metrics

By default, `Learner` will track only the loss function. You can track other metric with the [`Metrics`](#) callback. See also [`Metric`](#), [`AbstractMetric`](#).

## Hyperparameter scheduling

The [`Scheduler`](#) callback takes care of hyperparameter scheduling. See the [Hyperparameter scheduling tutorial](tutorials/hyperparameters.md) and also [`Scheduler`](#), [`Schedule`](#), [`HyperParameter`](#).

## Logging

For logging, use the logging callbacks:

- [`LogMetrics`](#)
- [`LogHyperParams`](#)
- [`LogHistograms`](#)

They each can have multiple logging backends, but right now the only one implemented in *FluxTraining.jl* is [`TensorBoardBackend`](#). See also [`LoggerBackend`](#), [`log_to`](#), and [`Loggables.Loggable`](#).

There is also an external package [Wandb.jl](https://github.com/avik-pal/Wandb.jl) that implements a logging backend for [Weights&Biases](www.wandb.ai).

## Checkpointing

Use the [`Checkpointer`](#) callback to create model checkpoints after every epoch.

## Early Stopping

Use [`EarlyStopping`](#) to stop when a stopping criterion is met. Supports all criteria in [EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl).
