
module Phases

abstract type Phase end
abstract type AbstractTrainingPhase <: Phase end

struct TrainingPhase <: Phase end
struct ValidationPhase <: Phase end
struct TestPhase <: Phase end


export
    Phase,
    TrainingPhase,
    ValidationPhase,
    TestPhase
end # module

using .Phases
