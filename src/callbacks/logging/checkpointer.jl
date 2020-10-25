

"""
    Checkpointer(folder)

Saves `learner.model` to `folder` after every [`AbstractTrainingPhase`].

Use `FluxTraining.`[`loadmodel`](#) to load a model.
"""
struct Checkpointer <: Callback
    folder
    function Checkpointer(folder)
        mkpath(folder)
        return new(folder)
    end
end


stateaccess(::Checkpointer) = (
    model = Read(),
    cbstate = (metricsepoch = Read(), history = Read())
)

function on(::EpochEnd, phase::AbstractTrainingPhase, checkpointer::Checkpointer, learner)
    loss = last(learner.cbstate.metricsepoch[phase], :Loss)[2]
    epoch = learner.cbstate.history.epochs
    filename = "checkpoint_epoch_$(lpad(string(epoch), 3, '0'))_loss_$loss.bson"
    savemodel(learner.model, joinpath(checkpointer.folder, filename))
end


# TODO: replace with JLD2?
function savemodel(model, path)
    @save path model = cpu(model)
end

"""
    loadmodel(path)

Loads a model that was saved to `path` using `FluxTraining.`[`savemodel`](#).
"""
function loadmodel(path)
    @load path model
    return model
end
