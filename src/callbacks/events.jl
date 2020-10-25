

"""
    Events
"""
module Events

"""
Abstract type for events that callbacks can hook into
"""
abstract type Event end

"""
Supertype for events that are called within `Zygote.gradient`.
They need to be handled differently because try/catch is not
supported by Zygote's compiler.
"""
abstract type GradEvent <: Event end

"""
    Init <: Event

Called once when the learner is created/the callback is added.
"""
struct Init <: Event end

struct FitBegin <: Event end
struct FitEnd <: Event end

struct EpochBegin <: Event end
struct EpochEnd <: Event end

struct BatchBegin <: Event end
struct BatchEnd <: Event end

"""Called between calculating `y_pred` and calculating loss"""
struct LossBegin <: GradEvent end
"""Called between calculating loss and calculating gradients"""
struct BackwardBegin <: GradEvent end
"""Called between calculating gradients and updating parameters"""
struct BackwardEnd <: Event end


export
    # asbtract
    Event, GradEvent,
    # concrete
    Init,
    FitBegin, FitEnd,
    EpochBegin, EpochEnd,
    BatchBegin, BatchEnd,
    LossBegin,
    BackwardBegin, BackwardEnd

end # module


using .Events
