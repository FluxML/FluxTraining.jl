
"""
    Phases
"""
module Phases

"""
    abstract type Phase

Abstract supertype for all phases. See `subtypes(FluxTraining.Phase)`.
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
    TrainingPhase()

A regular validation phase. It iterates over batches
in `learner.data.validation` and performs a forward pass.

Throws the following events: [`EpochBegin`](#), [`StepBegin`](#),
[`LossBegin`](#), [`StepEnd`](#), [`EpochEnd`](#).
"""
struct ValidationPhase <: Phase end
struct TestPhase <: Phase end


export
    Phase,
    AbstractTrainingPhase,
    TrainingPhase,
    ValidationPhase,
    TestPhase
end # module

using .Phases
