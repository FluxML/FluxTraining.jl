

struct Callbacks
    cbs::Vector
    executor::CallbackExecutor
    graph::SimpleDiGraph
end
Callbacks(cbs, executor = LinearExecutor()) = Callbacks(cbs, executor, callbackgraph(cbs))


getmetrics(cbs::Callbacks) = [cb for cb in cbs.cbs if cb isa AbstractMetric]
getloss(cbs::Callbacks) = only([cb for cb in cbs.cbs if cb isa AverageLoss])

"""
    $TYPEDEF

Stores data of the last processed batch.

$FIELDS

(!) If used in callbacks, some fields may be `nothing` as
they are reset after every step.
"""
# TODO: maybe make this dependent on the current `Phase`?
@with_kw mutable struct BatchState
    xs = nothing
    ys = nothing
    ŷs = nothing
    loss = nothing
    grads = nothing
end



"""
    Learner(model, data, opt, lossfn)

Central object for training that holds all necessary state.

$(TYPEDFIELDS)

# Fields
- `device`: device to train on, usually `Flux.cpu` or `Flux.gpu`
- `params`: model parameters, i.e. `Flux.params(model)`
- `phase::`[`AbstractFittingPhase`](@ref): current fitting phase
- `metrics::AbstractVector{`[`AbstractMetric`](@ref)`}`: metric callbacks to
    evaluate performance during training
- `config`
- `callbacks::AbstractVector{`[`AbstractCallback`](@ref)`}`: callbacks for training loop
- `recorder::`[`Recorder`](@ref): special callback that records metrics and hyperparameters
- `scheduler::`[`Scheduler`](@ref): special callback that records metrics and hyperparameters
- `batch::`[`BatchState`](@ref): Holds training state during batch
"""
@with_kw mutable struct Learner
    model
    data
    opt
    lossfn
    params
    batch::BatchState
    callbacks::Callbacks
    cbstate::Dict
end


function Learner(
        model, data, opt, lossfn;
        callbacks = [], metrics = [], schedule = Schedules(),
        usedefaultcallbacks = true, config = Dict(), cbexecutor = LinearExecutor()
    )
    if usedefaultcallbacks
        callbacks = vcat(defaultcallbacks(), callbacks)
    end
    callbacks = vcat(callbacks, metrics)
    Callbacks(callbacks)

    return Learner(
        model, data, opt, lossfn,
        Flux.params(model),
        BatchState(),
        Callbacks(callbacks, cbexecutor),
        Dict{Symbol, Any}())
end


defaultcallbacks()::Vector{AbstractCallback} = [
    ProgressBarLogger(),
    MetricsLogger(),
    StopOnNaNLoss(),
    Recorder(),
    AverageLoss(),
]


#  Callback handling
handle(event, learner, phase) = handle(learner.callbacks.executor, event, phase, learner)


# Other

getdataloader(phase::AbstractTrainingPhase, learner) = learner.data[1]
getdataloader(phase::ValidationPhase, learner) = learner.data[2]


function model!(learner, model)
    learner.model = model
    learner.params = Flux.params(model)
end


numsteps(learner::Protected, phase) = numsteps(getfield(learner, :data), phase)
numsteps(learner, phase) = length(getdataloader(phase, learner))


# TODO: fix

#=
function setschedule!(learner, schedule)
    learner.scheduler = ParamScheduler(
        delayschedule(schedule, learner.recorder.epoch)
    )
end
=#


# TODO: fix
#=
function numsteps(
        learner::Learner,
        phase::AbstractFittingPhase = learner.state.phase)
    return length(getdataloader(phase, learner))
end
numsteps(p::Protected{Learner}, phase) = numsteps(getfield(p, :data), phase)
=#

# TODO: move to Callback

#=
function artifactpath(learner)
    p = joinpath(pwd(), ".artifacts", string(learner.runid))
    if !isdir(p)
        mkpath(p)
    end
    return p
end
=#
