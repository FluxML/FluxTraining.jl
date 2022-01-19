

"""
    Checkpointer(folder)

Saves `learner.model` to `folder` after every [`AbstractTrainingPhase`](#).

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
    epoch = learner.cbstate.history[phase].epochs
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


# Top K Checkpointer
struct TopKCheckpointer <: Callback
    folder
    topk::Int
    metricname::Symbol
    ascending::Bool
    sortedtopkmetric::Vector
    filenames::Vector{String}

    function TopKCheckpointer(folder; k::Int = 5, metricname::Symbol = :Loss, ascending::Bool = false)
        @assert k != 0 "TopKCheckpointing with k = 0 is meaningless"
        if k < 0
            Base.printstyled("[TopKCheckpointer] k = $k < 0 => All checkpoints are retained"; color = :yellow)
        end
        mkpath(folder)
        return new(folder, k, metricname, ascending, fill(NaN, k), Vector{String}(undef, k))
    end
end

stateaccess(::TopKCheckpointer) = (
    model = Read(),
    cbstate = (metricsepoch = Read(), history = Read())
)

function on(::EpochEnd, phase::AbstractValidationPhase, checkpointer::TopKCheckpointer, learner)
    metric = last(learner.cbstate.metricsepoch[phase], checkpointer.metricname)[2]
    epoch = learner.cbstate.history[phase].epochs

    filename = joinpath(checkpointer.folder, "checkpoint_epoch_$(lpad(string(epoch), 3, '0'))_$(checkpointer.metricname)_$metric.bson")

    if epoch <= checkpointer.topk
        checkpointer.sortedtopkmetric[epoch] = metric
        checkpointer.filenames[epoch] = filename

        if epoch == checkpointer.topk
            sort!(checkpointer.sortedtopkmetric; rev = !checkpointer.ascending)
        end
    else
        idx = searchsortedfirst(checkpointer.sortedtopkmetric, metric; rev = !checkpointer.ascending)
        idx == 1 && return
        rm(checkpointer.filenames[1]; force = true)
        if idx != 2
            checkpointer.sortedtopkmetric[1:idx - 2] .= checkpointer.sortedtopkmetric[2:idx - 1]
            checkpointer.filenames[1:idx - 2] .= checkpointer.filenames[2:idx - 1]
        end
        checkpointer.sortedtopkmetric[idx - 1] = metric
        checkpointer.filenames[idx - 1] = filename
    end

    savemodel(learner.model, filename)
    return
end