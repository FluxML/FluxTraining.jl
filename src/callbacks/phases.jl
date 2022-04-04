
module Phases

"""
    abstract type Phase

Abstract supertype for all phases. See `subtypes(FluxTraining.Phase)`.
A `Phase` is used in dispatch for training loop functions [`step!`](#)
and [`epoch!`](#) as well as in [`Callback`](#) handler methods [`on`](#).
"""
abstract type Phase end

"""
    abstract type AbstractTrainingPhase <: Phase

An abstract type for phases where parameter updates
are being made. This exists so callbacks can dispatch on it and work
with custom training phases.

The default implementation for supervised tasks is [`TrainingPhase`](#).
"""
abstract type AbstractTrainingPhase <: Phase end

"""
    TrainingPhase() <: AbstractTrainingPhase

A regular training phase for supervised learning. It iterates over batches
in `learner.data.training` and updates the model parameters
using `learner.optim` after calculating the gradients.

Throws the following events in this order:

- [`EpochBegin`](#) when an epoch starts,
- [`StepBegin`](#) when a step starts,
- [`LossBegin`](#) after the forward pass but before loss calculation,
- [`BackwardBegin`](#) after loss calculation but before backward pass,
- [`BackwardEnd`](#) after the bacward pass but before the optimization
    step,
- [`StepEnd`](#) when a step ends; and
- [`EpochEnd`](#) when an epoch ends

It writes the following step state to `learner.state`, grouped by the
event from which on it is available.

- `StepBegin`:
    - `xs` and `ys`: encoded input and target (batch)
- `LossBegin`:
    - `ŷs`: model output
- `BackwardBegin`:
    - `loss`: loss
- `BackwardEnd`:
    - `grads`: calculated gradients
"""
struct TrainingPhase <: AbstractTrainingPhase end

"""
    abstract type AbstractValidationPhase <: Phase

An abstract type for phases where no parameter updates
are being made. This exists so callbacks can dispatch on it and work
with custom validation phases.

The default implementation for supervised tasks is [`ValidationPhase`](#).
"""
abstract type AbstractValidationPhase <: Phase end

"""
    ValidationPhase()

A regular validation phase. It iterates over batches
in `learner.data.validation` and performs a forward pass.

Throws the following events: [`EpochBegin`](#), [`StepBegin`](#),
[`LossBegin`](#), [`StepEnd`](#), [`EpochEnd`](#).

Throws the following events in this order:

- [`EpochBegin`](#) when an epoch starts,
- [`StepBegin`](#) when a step starts,
- [`LossBegin`](#) after the forward pass but before loss calculation,
- [`StepEnd`](#) when a step ends; and
- [`EpochEnd`](#) when an epoch ends

It writes the following step state to `learner.state`, grouped by the
event from which on it is available.

- `StepBegin`:
    - `xs` and `ys`: encoded input and target (batch)
- `LossBegin`:
    - `ŷs`: model output
- `StepEnd`:
    - `loss`: loss
"""
struct ValidationPhase <: AbstractValidationPhase end


export
    Phase,
    AbstractTrainingPhase,
    TrainingPhase,
    AbstractValidationPhase,
    ValidationPhase
end # module

using .Phases
