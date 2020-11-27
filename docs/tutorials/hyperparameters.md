# Hyperparameter scheduling

When training neural networks, you often have to tune hyperparameters.

In *FluxTraining.jl* the following definition is used:

> *A hyperparameter is any state that influences the training and is not a parameter of the model.*

Common hyperparameters to worry about are the learning rate, batch size and regularization strength.

In recent years, it has also become common practice to schedule some hyperparameters. The cyclical learning rate schedule introduced in [L. Smith 2015](https://arxiv.org/abs/1506.01186), for example, changes the learning rate every step to speed up convergence.

*FluxTraining.jl* provides an extensible interface for hyperparameter scheduling that is not restricted to optimizer hyperparameters as in many other training frameworks. To use it, you have to create a [`Scheduler`](#), a callback that can be passed to a [`Learner`](#).

`Scheduler`'s constructor takes pairs of hyperparameter types and associated schedules.

As an example

- [`LearningRate`](#) is a hyperparameter type representing the optimizer's step size; and
- `schedule = Schedule([0, 10], [0.01, 0.001])` represents a linear scheduling over 10 epochs from `0.01` to `0.001` 

We can create the callback scheduling the learning rate according to `Scheduler(LearningRate => schedule)`.

`Schedule`s are built around [*Animations.jl*](https://jkrumbiegel.github.io/Animations.jl/dev/). See that package's documentation or [`Schedule`](#)'s for more details on how to construct them. 

### One-cycle learning rate

Let's define a `Schedule` that follows the above-mentioned cyclical learning rate schedule.

The idea is to start with a small learning rate, gradually increase it, and then slowly decrease it again.

For example, we could start with a learning rate of 0.01, increase it to 0.1 over 3 epochs, and then down to 0.001 over 7 epochs. Let's also use cosine annealing, a common practice that makes sure the values are interpolated more smoothly.

In code, that looks like this:

```julia
using Animations: sineio  # for cosine annealing

schedule = Schedule(
    [0, 3, 10],          # the time steps (in epochs)
    [0.01, 0.1, 0.001],  # the valus at the time steps
    sineio(),            # the annealing function
)

learner = model(model, data, opt, lossfn, Scheduler(LearningRate => schedule))
```

For convenience, you can also use the [`onecycle`](#) helper to create this `Schedule`.

## Extending

You can create and schedule your own hyperparameters.

To do this, you will need to define

- a type for your hyperparameter, e.g. `abstract type MyParam <: HyperParameter end`,
- how to set the hyperparameter by implementing [`sethyperparameter!`](#)`(learner, ::Type{MyParam}, value)`
- what state needs to be accessed to set the hyperparameter by implementing [`stateaccess`](#)`(::Type{MyParam})`. See [custom callbacks](../callbacks/custom.md) for more info on why this is needed.

!!! info "Kinds of hyperparameters"

    Hyperparameters don't need to belong to the optimizer! For example, you could create a hyperparameter for batch size. That is not implemented here because this package is agnostic of the data iterators and the implementation would differ for every type of iterator.
