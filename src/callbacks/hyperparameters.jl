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
    sethyperparameter!(learner, H, value)

Sets hyperparameter `H` to `value` on `learner`.
"""
function sethyperparameter! end

"""
    stateaccess(::Type{HyperParameter})

Defines what `Learner` state is accessed when calling
`sethyperparameter!` and `gethyperparameter`
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
sethyperparameter!(learner, ::Type{LearningRate}, value) =
    setlearningrate!(learner.optimizer, value)

function setlearningrate!(optimizer, value)
    optimizer.eta = value
end
