# Getting started

Let's look at a simple training example. In *FluxTraining.jl*, a [`Learner`](#) holds all state necessary for training. To get started, you need

- a [model](/doc/docs/background/model.md)
- training and validation [data iterators](/doc/docs/background/dataiterator.md)
- a [loss function](/doc/docs/background/lossfunction.md); and
- an [optimizer](/doc/docs/background/optimizer.md)

First we define the necessary pieces:

```julia
model = ...
traindata, valdata = ...
lossfn = Flux.Losses.mse
opt = Flux.ADAM(0.01)
```

Then we construct a [`Learner`](#):

```julia
learner = Learner(model, lossfn; data=(traindata, valdata))
```

And train for 10 epochs:

```julia
fit!(learner, 10)
```
