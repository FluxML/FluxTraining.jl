
abstract type AbstractCallback end
abstract type SafeCallback <: AbstractCallback end
const Callback = SafeCallback
abstract type UnsafeCallback <: AbstractCallback end


abstract type AbstractMetric <: Callback end
abstract type AbstractLogger <: Callback end


canwrite(::SafeCallback) = nothing

order(c::Type{<:AbstractCallback}) = 0
order(c::Type{<:AbstractMetric}) = -100
order(c::Type{<:AbstractLogger}) = 100
order(c::T) where T<:AbstractCallback = order(T)


# Training control flow

abstract type FitException <: Exception end
struct CancelBatchException <: FitException
    msg::String
end
struct CancelEpochException <: FitException
    msg::String
end
struct CancelFittingException <: FitException
    msg::String
end


# Callback hook

"""
    on(event::FitEvent, phase::AbstractFittingPhase, callback::AbstractCallback, learner)

Handle `event` with `callback`. Can dispatch on an `AbstractFittingPhase` and
receives `learner` as an additional argument.

If not overwritten with a more specific method, does nothing.

To see events which an `AbstractCallback` handles, use

    `methods(Training.on, (Any, Any, MyCallbackType, Any)`
"""
on(::FitEvent, ::AbstractFittingPhase, ::AbstractCallback, learner) = return

_on(e, p, cb, learner) = on(e, p, cb, learner)
#_on(e, p, cb::SafeCallback, learner) = on(e, p, cb, protect(learner, canwrite(cb)))
_on(e, p, cb::SafeCallback, learner) = on(e, p, cb, learner)
