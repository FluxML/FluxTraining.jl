
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

An abstract phase representing phases where parameter updates
are being made. This exists so callbacks can dispatch on it and work
with custom training phases.

The default implementation is [`TrainingPhase`](#).
"""
abstract type AbstractTrainingPhase <: Phase end

struct TrainingPhase <: AbstractTrainingPhase end
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
