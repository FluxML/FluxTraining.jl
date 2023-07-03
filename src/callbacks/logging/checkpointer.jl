

"""
    Checkpointer(folder)

Saves `learner.model` to `folder` after every [`AbstractTrainingPhase`](#).
If `keep_top_k` is provided, only the best k models (by smallest training loss) and the latest model are kept.

Use `FluxTraining.`[`loadmodel`](#) to load a model.
"""
struct Checkpointer <: Callback
    folder
    keep_top_k::Union{Int, Nothing}
    top_k::PriorityQueue{<:AbstractString, <:Real}
    function Checkpointer(folder; keep_top_k::Union{Int, Nothing}=nothing)
        mkpath(folder)
        # We want to pop the worst checkpoints, i.e. with highest loss.
        # Therefore, we must reverse Julia's PriorityQueue ordering, which defaults to lowest first.
        return new(folder, keep_top_k, PriorityQueue{String, Float64}(Base.Order.Reverse))
    end
end


stateaccess(::Checkpointer) = (
    model = Read(),
    cbstate = (metricsepoch = Read(), history = Read())
)

function on(::EpochEnd, phase::AbstractTrainingPhase, checkpointer::Checkpointer, learner)
    loss = last(learner.cbstate.metricsepoch[phase], :Loss)[2]
    epoch = learner.cbstate.history[phase].epochs
    filename = "checkpoint_epoch_$(lpad(string(epoch), 3, '0'))_loss_$loss.bson"

    savemodel(learner.model, joinpath(checkpointer.folder, filename))

    if !isnothing(checkpointer.keep_top_k)
        process_top_k_checkpoints(checkpointer, filename, loss)
    end
end


"Makes sure only the best k and the latest checkpoints are kept on disk."
function process_top_k_checkpoints(checkpointer::Checkpointer, new_checkpoint::String, new_loss::Real)
    # Note that priority queue may have k+1 elements, also tracking the most recent checkpoint.
    @assert length(checkpointer.top_k) <= checkpointer.keep_top_k+1
    if length(checkpointer.top_k) == checkpointer.keep_top_k+1  # if most recent checkpoint was worst, remove it
        most_recent_checkpoint = dequeue!(checkpointer.top_k)
        rm(joinpath(checkpointer.folder, most_recent_checkpoint);
           force=true)  # force=true so that an error doesn't cancel the training
    end
    enqueue!(checkpointer.top_k, new_checkpoint=>new_loss)
    if (length(checkpointer.top_k) > checkpointer.keep_top_k  # We try to shorten the queue...
        && peek(checkpointer.top_k)[1] != new_checkpoint)     # ...but don't remove new checkpoint even if it's the worst,
                                                              # potentially creating (k+1) elements.
        worst_checkpoint_that_is_not_new = dequeue!(checkpointer.top_k)
        rm(joinpath(checkpointer.folder, worst_checkpoint_that_is_not_new);
           force=true)  # force=true so that an error doesn't cancel the training
    end
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
