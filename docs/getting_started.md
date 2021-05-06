# Getting started

*FluxTraining.jl* focuses on a powerful and customizable training loop, and as such models and datasets are not included. It powers the training in [FastAI.jl](https://github.com/FluxML/FastAI.jl), which covers every part of the deep learning pipeline.

---


Let's look at a simple training example. In *FluxTraining.jl*, a [`Learner`](#) holds all state necessary for training. To get started, you need

- a *model*
- training and validation *data iterators*
- a *loss function*; and
- an *optimizer*

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
