# CheckpointCondition
abstract type CheckpointCondition end

struct CheckpointAny <: CheckpointCondition end
mutable struct CheckpointLowest <: CheckpointCondition
    lowest::Real
end
CheckpointLowest() = CheckpointLowest(Inf)

(::CheckpointAny)(loss) = true

function (checklowest::CheckpointLowest)(loss)
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


# TODO: refactor to use `Metrics`
function on(::EpochEnd, ::ValidationPhase, checkpointer::Checkpointer, learner)
    loss = epochvalue(getloss(learner.callbacks))
    if checkpointer.condition(loss)
        if checkpointer.deleteprevious
            previousfiles = glob("model-chckpnt-E*", artifactpath(learner))
            foreach(rm, previousfiles)
        end
        epochs = learner.cbstate.history.epochs
        filename = "model-chckpnt-E$epochs-L$loss.bson"
        path = joinpath(artifactpath(learner), filename)
        savemodel(learner.model, path)
    end
end

stateaccess(::Checkpointer) = (callbacks = Read(), cbstate = (history = Read()))
