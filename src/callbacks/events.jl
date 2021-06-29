"""
    module Events

Provides the abstract [`Event`](#) type and concrete event types.

- [`Init`](#) is called once for every callback. Use for initialization code
  that needs access to some `Learner` state.

Events in [`TrainingPhase`](#) and [`ValidationPhase`](#):

- [`EpochBegin`](#) and [`EpochEnd`](#), called at the beginning and end of each
  epoch.
- [`StepBegin`](#) and [`StepEnd`](#), called at the beginning and end of each
  batch.
- [`LossBegin`](#), called after the forward pass but before the loss calculation.

[`TrainingPhase`](#) only:

- [`BackwardBegin`](#), called after forward pass and loss calculation but before gradient
  calculation.
- [`BackwardEnd`](#), called after gradient calculation but before parameter update.

"""
module Events

"""
    abstract type Event

Abstract type for events that callbacks can hook into
"""
abstract type Event end

"""
    Init <: Event

Called once when the learner is created/the callback is added.

Hook into this for callback initialization that depends on the
[`Learner`](#)'s state.
"""
struct Init <: Event end

"""
    EpochBegin()

[`Event`](#) called at the beginning of an epoch.
"""
struct EpochBegin <: Event end

"""
    EpochEnd()

[`Event`](#) called at the end of an epoch.
"""
struct EpochEnd <: Event end

"""
    StepBegin()

[`Event`](#) called at the beginning of a batch.
"""
struct StepBegin <: Event end

"""
    StepEnd()

[`Event`](#) called at the end of a batch.
"""
struct StepEnd <: Event end

"""
    LossBegin()

[`Event`](#) called between calculating `y_pred` and calculating loss
"""
struct LossBegin <: Event end

"""
    BackwardBegin()

[`Event`](#) called between calculating loss and calculating gradients
"""
struct BackwardBegin <: Event end

"""
    BackwardEnd()

[`Event`](#) called between calculating gradients and updating parameters.
"""
struct BackwardEnd <: Event end


export
    # abstract
    Event,
    # concrete
    Init,
    EpochBegin, EpochEnd,
    StepBegin, StepEnd,
    LossBegin,
    BackwardBegin, BackwardEnd

end # module


using .Events
