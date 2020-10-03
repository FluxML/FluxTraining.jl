
"""
    Phases
"""
module Phases

abstract type Phase end
abstract type AbstractTrainingPhase <: Phase end

struct TrainingPhase <: AbstractTrainingPhase end


struct ValidationPhase <: Phase end
struct TestPhase <: Phase end
struct InitializationPhase <: Phase end
struct CleanupPhase <: Phase end


export
    Phase,
    AbstractTrainingPhase,
    TrainingPhase,
    ValidationPhase,
    TestPhase,
    InitializationPhase,
    CleanupPhase
end # module

using .Phases
