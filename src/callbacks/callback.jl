"""
    Abstract Callback

Abstract supertype for all callbacks

# Callback interface

- `order(cb) = 0`

"""
abstract type AbstractCallback end
abstract type AbstractMetric <: AbstractCallback end
abstract type AbstractLogger <: AbstractCallback end

order(c::Type{<:AbstractCallback}) = 0
order(c::Type{<:AbstractMetric}) = -100
order(c::Type{<:AbstractLogger}) = 100
order(c::T) where T<:AbstractCallback = order(T)


"""
    Phases
"""
module Phases

abstract type AbstractFittingPhase end
abstract type AbstractTrainingPhase <: AbstractFittingPhase end

struct TrainingPhase <: AbstractTrainingPhase end


struct ValidationPhase <: AbstractFittingPhase end
struct TestPhase <: AbstractFittingPhase end
struct InitializationPhase <: AbstractFittingPhase end
struct CleanupPhase <: AbstractFittingPhase end


export
    AbstractFittingPhase,
    AbstractTrainingPhase, 
    TrainingPhase,
    ValidationPhase,
    TestPhase,
    InitializationPhase,
    CleanupPhase
end # module

using .Phases


"""
    Events
"""
module Events

"""
Abstract type for events that callbacks can hook into
"""
abstract type FitEvent end

"""
Supertype for events that are called within `Zygote.gradient`.
They need to be handled differently because try/catch is not
supported by Zygote's compiler.
"""
abstract type GradEvent <: FitEvent end

struct FitBegin <: FitEvent end
struct FitEnd <: FitEvent end

struct EpochBegin <: FitEvent end
struct EpochEnd <: FitEvent end

struct BatchBegin <: FitEvent end
struct BatchEnd <: FitEvent end

"""Called between calculating `y_pred` and calculating loss"""
struct LossBegin <: GradEvent end
"""Called between calculating loss and calculating gradients"""
struct BackwardBegin <: GradEvent end
"""Called between calculating gradients and updating parameters"""
struct BackwardEnd <: FitEvent end


export 
    # asbtract
    FitEvent, GradEvent,
    # concrete
    FitBegin, FitEnd,
    EpochBegin, EpochEnd,
    BatchBegin, BatchEnd,
    LossBegin,
    BackwardBegin, BackwardEnd

end # module


using .Events


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


# CallbackHandler

struct CallbackHandler
    learner
    callbacks::Vector{AbstractCallback}
    errored::Set{AbstractCallback}
    CallbackHandler(learner, callbacks) = new(
        learner,
        sort!(callbacks, by=cb -> order(typeof(cb))),
        Set{AbstractCallback}(),
    )
end

function (handler::CallbackHandler)(event::FitEvent)
    foreach(handler.callbacks) do callback
        if !(callback in handler.errored)
            try
                on(event, handler.learner.phase, callback, handler.learner)
            catch e
                if e isa Union{FitException, InterruptException} || callback isa AbstractMetric
                    rethrow()
                else
                    @error "Callback $callback threw an unexpected error, disabling it." error=e
                    push!(handler.errored, callback)
                end
            end
        end
    end
end

function (handler::CallbackHandler)(event::GradEvent)
    foreach(handler.callbacks) do callback
        on(event, handler.learner.phase, callback, handler.learner)
    end
end