
"""
    Phases
"""
module Phases

abstract type Phase end
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
