

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
