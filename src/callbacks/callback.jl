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


# Phases

"""
    AbstractFittingPhase
"""
abstract type AbstractFittingPhase end
abstract type AbstractTrainingPhase <: AbstractFittingPhase end

struct TrainingPhase <: AbstractTrainingPhase end


struct ValidationPhase <: AbstractFittingPhase end
struct TestPhase <: AbstractFittingPhase end
struct InitializationPhase <: AbstractFittingPhase end
struct CleanupPhase <: AbstractFittingPhase end

# Events

"""
Abstract type for events that callbacks can hook into
"""
abstract type FitEvent end

struct TrainingBegin <: FitEvent end
struct TrainingEnd <: FitEvent end

struct EpochBegin <: FitEvent end
struct EpochEnd <: FitEvent end

struct BatchBegin <: FitEvent end
struct BatchEnd <: FitEvent end

"""Called between calculating `y_pred` and calculating loss"""
struct LossBegin <: FitEvent end
"""Called between calculating loss and calculating gradients"""
struct BackwardBegin <: FitEvent end
"""Called between calculating gradients and updating parameters"""
struct BackwardEnd <: FitEvent end

struct Initialize <: FitEvent end
struct Cleanup <: FitEvent end


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
    callbacks
    CallbackHandler(learner, callbacks) = new(
        learner,
        sort!(callbacks, by=cb -> order(typeof(cb)))
    )
end

function (ch::CallbackHandler)(event::FitEvent)
    foreach(ch.callbacks) do cb
        on(event, ch.learner.phase, cb, ch.learner)
    end
end
