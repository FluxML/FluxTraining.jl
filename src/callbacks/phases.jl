
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
