
module Phases

"""
    abstract type Phase

Abstract supertype for all phases. See `subtypes(FluxTraining.Phase)`.
A `Phase` is used in dispatch for training loop functions [`step!`](#) and [`epoch!`](#)
as well as in callback handler methods [`on`](#).
"""
abstract type Phase end

"""
    abstract type AbstractTrainingPhase <: Phase

An abstract type for phases where parameter updates
are being made. This exists so callbacks can dispatch on it and work
with custom training phases.

The default implementation is [`TrainingPhase`](#).
"""
abstract type AbstractTrainingPhase <: Phase end

"""
    TrainingPhase()

A regular training phase. It iterates over batches
in `learner.data.training` and updates the model parameters
using `learner.optim` after calculating the gradients.

Throws the following events: [`EpochBegin`](#), [`StepBegin`](#),
[`LossBegin`](#), [`BackwardBegin`](#), [`BackwardEnd`](#), [`StepEnd`](#),
[`EpochEnd`](#).
"""
struct TrainingPhase <: AbstractTrainingPhase end

"""
    abstract type AbstractValidationPhase <: Phase

An abstract type for phases where no parameter updates
are being made. This exists so callbacks can dispatch on it and work
with custom validation phases.

The default implementation is [`ValidationPhase`](#).
"""
abstract type AbstractValidationPhase <: Phase end

"""
    ValidationPhase()

A regular validation phase. It iterates over batches
in `learner.data.validation` and performs a forward pass.

Throws the following events: [`EpochBegin`](#), [`StepBegin`](#),
[`LossBegin`](#), [`StepEnd`](#), [`EpochEnd`](#).
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
