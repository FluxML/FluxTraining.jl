"""
Abstract type for events that callbacks can hook into
"""
abstract type FitEvent end

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


"""
    on(::FitEvent, ::AbstractFittingPhase, ::AbstractCallback, learner)

Handle `event` with `callback`. Can dispatch on an `AbstractFittingPhase` and
receives `learner` as an additional argument.

If not overwritten with a more specific method, does nothing

To see events which an `AbstractCallback` handles, use 

    `methods(Training.on, (Any, Any, MyCallback, Any)`
"""
on(::FitEvent, ::AbstractFittingPhase, ::AbstractCallback, learner) = return
