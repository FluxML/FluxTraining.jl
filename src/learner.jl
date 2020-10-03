

mutable struct Callbacks
    cbs::Vector
    executor::CallbackExecutor
    graph::SimpleDiGraph
    initialized::Bool
end
Callbacks(cbs, executor = LinearExecutor()) = Callbacks(cbs, executor, callbackgraph(cbs), false)


getmetrics(cbs::Callbacks) = [cb for cb in cbs.cbs if cb isa AbstractMetric]
getloss(cbs::Callbacks) = only([cb for cb in cbs.cbs if cb isa Loss])

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
    Learner(model, (traindata, valdata), opt, lossfn; kwargs...)

Holds and coordinates all state of the training.

"""
mutable struct Learner
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
        callbacks = [], metrics = [], schedules = Dict(),
        usedefaultcallbacks = true, config = Dict(), cbexecutor = LinearExecutor()
    )
    if usedefaultcallbacks
        callbacks = vcat(defaultcallbacks(schedules = schedules), callbacks)
    end
    callbacks = vcat(callbacks, metrics)
    Callbacks(callbacks)

    return Learner(
        model, dataiters(data), opt, lossfn,
        Flux.params(model),
        BatchState(),
        Callbacks(callbacks, cbexecutor),
        Dict{Symbol, Any}())
end

Base.show(io::IO, learner::Learner) = print(io, "Learner()")

defaultcallbacks(; schedules = Dict())::Vector{AbstractCallback} = [
    ProgressBarLogger(),
    MetricsLogger(),
    StopOnNaNLoss(),
    Recorder(),
    Loss(),
    Scheduler(schedules),
]


#  Callback handling
handle(event, learner, phase) = handle(learner.callbacks.executor, event, phase, learner)


# Other

getdataiter(phase::AbstractTrainingPhase, learner) = learner.data.training
getdataiter(phase::ValidationPhase, learner) = learner.data.validation


function model!(learner, model)
    learner.model = model
    learner.params = Flux.params(model)
end


numsteps(learner::Protected, phase) = numsteps(getfield(learner, :data), phase)
numsteps(learner, phase) = length(getdataloader(phase, learner))

hascallback(learner::Protected, T) = hascallback(getfield(learner, :data), phase)
hascallback(learner, T) = any(C <: T for C in typeof.(learner.callbacks.cbs))


dataiters(t::Tuple) = dataiters(t...)
dataiters(train, val, test = nothing) = (training = train, validation = val, test = test)
dataiters(t::NamedTuple) = keys(t) == (:training, :validation, :test) ? t : error("Wrong keys.")
# TODO: fix

#=
function setschedule!(learner, schedule)
    learner.scheduler = ParamScheduler(
        delayschedule(schedule, learner.recorder.epoch)
    )
end
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
