using BSON: @load, @save
using Glob

# CheckpointCondition
abstract type CheckpointCondition end

struct CheckpointAny <: CheckpointCondition end
mutable struct CheckpointLowest <: CheckpointCondition
    lowest::Real
end
CheckpointLowest() = CheckpointLowest(Inf)

(::CheckpointCondition)(learner) = error()
(::CheckpointAny)(learner) = true

function (checklowest::CheckpointLowest)(learner)
    loss = value(learner.metrics[1])
    cond = loss < checklowest.lowest
    if cond
        checklowest.lowest = loss
    end
    return cond
end

# Checkpointer
struct Checkpointer <: AbstractCallback
    condition::CheckpointCondition
    deleteprevious::Bool
end
Checkpointer(condition = CheckpointLowest(); deleteprevious = false) = Checkpointer(
    condition, deleteprevious)


function on(::EpochEnd, ::ValidationPhase, checkpointer::Checkpointer, learner)
    if checkpointer.condition(learner)
        if checkpointer.deleteprevious
            previousfiles = glob("model-chckpnt-E*", artifactpath(learner))
            foreach(rm, previousfiles)
        end
        loss = value(learner.metrics[1])
        filename = "model-chckpnt-E$(learner.recorder.epoch)-L$loss.bson"
        path = joinpath(artifactpath(learner), filename)
        savemodel(learner.model, path)
    end
end
