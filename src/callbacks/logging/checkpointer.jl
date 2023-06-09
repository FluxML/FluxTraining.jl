

"""
    Checkpointer(folder)

Saves `learner.model` to `folder` after every [`AbstractTrainingPhase`](#).
If `keep_top_k` is provided, only the best k models (by smallest training loss) and the latest model are kept.

Use `FluxTraining.`[`loadmodel`](#) to load a model.
"""
struct Checkpointer <: Callback
    folder
    keep_top_k::Union{Integer, Nothing}
    top_k::PriorityQueue{<:AbstractString, <:Real}
    function Checkpointer(folder; keep_top_k::Union{Integer, Nothing}=nothing)
        mkpath(folder)
        # notice that in Julia, the PriorityQueue gives you elements with the *lowest* priority first
        return new(folder, keep_top_k, PriorityQueue{String, Float64}(Base.Order.Reverse))
    end
end


stateaccess(::Checkpointer) = (
    model = Read(),
    cbstate = (metricsepoch = Read(), history = Read())
)

function on(::EpochEnd, phase::AbstractTrainingPhase, checkpointer::Checkpointer, learner)
    epoch = learner.cbstate.history[phase].epochs
    loss = last(learner.cbstate.metricsepoch[phase], :Loss)[2]
    filename = "checkpoint_epoch_$(lpad(string(epoch), 3, '0'))_loss_$loss.bson"

    savemodel(learner.model, joinpath(checkpointer.folder, filename))

    if !isnothing(checkpointer.keep_top_k)
        process_top_k_checkpoints(checkpointer, filename, loss)
    end
end

try_fs_remove(path::String) =
    try; Base.Filesystem.rm(path)
    catch e; @warn e; end


"Makes sure only the best k and the latest checkpoints are kept on disk."
function process_top_k_checkpoints(checkpointer::Checkpointer, new_checkpoint::String, new_loss::Real)
    # Note that priority queue may have k+1 elements, also tracking the latest model.
    @assert length(checkpointer.top_k) <= checkpointer.keep_top_k+1
    if length(checkpointer.top_k) > checkpointer.keep_top_k  # if previous model was not in top k
        worst_checkpoint = dequeue!(checkpointer.top_k)
        try_fs_remove(joinpath(checkpointer.folder, worst_checkpoint))
    end
    enqueue!(checkpointer.top_k, new_checkpoint=>new_loss)
    if length(checkpointer.top_k) > checkpointer.keep_top_k && peek(checkpointer.top_k)[1] != new_checkpoint
        worst_checkpoint = dequeue!(checkpointer.top_k)
        try_fs_remove(joinpath(checkpointer.folder, worst_checkpoint))
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
