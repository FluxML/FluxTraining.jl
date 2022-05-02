# Hyperparameter interface

"""
    HyperParameter{T}

A hyperparameter is any state that influences the
training and is not a parameter of the model.

Hyperparameters can be scheduled using the [`Scheduler`](#)
callback.
"""
abstract type HyperParameter{T} end

"""
    sethyperparameter!(learner, H, value) -> learner

Sets hyperparameter `H` to `value` on `learner`, returning
the modified learner.
"""
function sethyperparameter! end

"""
    stateaccess(::Type{HyperParameter})

Defines what `Learner` state is accessed when calling
`sethyperparameter!` and `gethyperparameter`. This is needed
so that [`Scheduler`](#) can access the state.
"""
stateaccess(::Type{HyperParameter}) = ()


# Implementations

"""
    abstract type LearningRate <: HyperParameter

Hyperparameter for the optimizer's learning rate.

See [`Scheduler`](#) and [hyperparameter scheduling](./docs/tutorials/hyperparameters.md).
"""
abstract type LearningRate <: HyperParameter{Float64} end

stateaccess(::Type{LearningRate}) = (optimizer = Write(),)

function sethyperparameter!(learner, ::Type{LearningRate}, value)
    learner.optimizer = setlearningrate!(learner.optimizer, value)
    return learner
end

function setlearningrate!(optimizer::Flux.Optimise.AbstractOptimiser, value)
    optimizer.eta = value
    optimizer
end

function setlearningrate!(optimizer, value)
    @set optimizer.eta = value
end
