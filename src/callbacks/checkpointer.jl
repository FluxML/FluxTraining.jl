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
    loss = value(learner.state.callbacks.loss)
    cond = loss < checklowest.lowest
    if cond
        checklowest.lowest = loss
    end
    return cond
end

# Checkpointer
struct Checkpointer <: SafeCallback
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
        loss = get(learner.state.history.epochmetrics[ValidationPhase()], :loss)[2][end]
        filename = "model-chckpnt-E$(learner.state.history.epochs)-L$loss.bson"
        path = joinpath(artifactpath(learner), filename)
        savemodel(learner.model, path)
    end
end
