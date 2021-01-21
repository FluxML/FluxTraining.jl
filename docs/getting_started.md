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
lossfn = Flux.Losses.mse
opt = Flux.ADAM(0.01)
```

Then we construct a [`Learner`](#):

```julia
learner = Learner(model, (traindata, valdata), opt, lossfn)
```

And train for 10 epochs:

```julia
fit!(learner, 10)
```
