# Getting started

In `FluxTraining`, a [`Learner`](#) holds all state necessary for training. To get started, you need
- a *model*
- training and validation *data iterators*
- a *loss function*; and
- an *optimizer*

Models and datasets are not included with FluxTraining.jl, so see [ecosystem](ecosystem.md) for supporting packages.

Let's look at a simple training example.

First we define the necessary pieces:

```julia
model = ...
traindata, valdata = ...
lossfn = Flux.mse
opt = Flux.ADAM(0.01)
```

Then we construct a [`Learner`](#):

```julia
learner = Learner(model, (traindata, valdata), lossfn, opt)
```

And train for 10 epochs:

```julia
fit!(learner, 10)
```

Most functionality in `FluxTraining.jl` is implemented as a [callback](../extending/extend_callbacks.md).

Callbacks can add all kinds of functionality by hooking into the training loop. For example, the following callbacks are used by default when you create a learner:

- [`ProgressBarLogger`](#) prints the progress of the current epoch
- [`MetricsLogger`](#) prints the metrics of the last epoch
- [`StopOnNaNLoss`](#) stops the training when a `NaN` loss is encountered