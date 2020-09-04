"""
    $TYPEDEF

Stores all callbacks of a `Learner`.

$TYPEDFIELDS
"""
@with_kw  mutable struct Callbacks
    loss::AverageLoss = AverageLoss()
    recorder::Recorder = Recorder()
    scheduler::ParamScheduler = ParamScheduler()
    metrics::Vector{AbstractMetric}
    other::Vector{AbstractCallback}
    all = sort!(
        [loss, metrics..., recorder, scheduler, other...],
        by = cb -> order(typeof(cb)))
end

Callbacks(callbacks, metrics) = Callbacks(other = callbacks, metrics = metrics)

"""
    $TYPEDEF

Stores data of the last processed batch.

$FIELDS

(!) If used in callbacks, some fields may be `nothing` as
they are reset after every step.
"""
@with_kw mutable struct BatchState
    xs = nothing
    ys = nothing
    ŷs = nothing
    loss = nothing
    grads = nothing
end


"""
    $TYPEDEF

Stores all state of a learner.

$TYPEDFIELDS

"""
@with_kw mutable struct LearnerState
    phase::AbstractFittingPhase = InitializationPhase()
    history::History = History()
    schedule::Schedules = Schedules()
    params
    batch::BatchState = BatchState()
    callbacks::Callbacks
    config::Dict = Dict()
    run::UUID = uuid4()
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
    # Flux model
    model
    # tuple of training and validation DataLoaders
    data
    # optimizer
    opt
    # loss function taking `lossfn(y_pred, y)`
    lossfn
    #
    state::LearnerState
end


function Learner(
        model, data, opt, lossfn;
        callbacks = [], metrics = [], schedule = Schedules(),
        usedefaultcallbacks = true, config = Dict()
    )
    if usedefaultcallbacks
        callbacks = vcat(defaultcallbacks(), callbacks)
    end

    state = LearnerState(
        schedule = schedule,
        callbacks = Callbacks(callbacks, metrics),
        params = params(model),
        config = config,
    )
    return Learner(model, data, opt, lossfn, state)
end


defaultcallbacks()::Vector{AbstractCallback} = [StopOnNaNLoss()]


#  Callback handling

function handle(event::FitEvent, learner::Learner)
    foreach(learner.state.callbacks.all) do callback
        on(event, learner.state.phase, callback, learner)
    end
end

# Other

getdataloader(phase::AbstractTrainingPhase, learner) = learner.data[1]
getdataloader(phase::ValidationPhase, learner) = learner.data[2]



# TOdo: fix

#=
function setschedule!(learner, schedule)
    learner.scheduler = ParamScheduler(
        delayschedule(schedule, learner.recorder.epoch)
    )
end
=#


function numsteps(
        learner::Learner,
        phase::AbstractFittingPhase = learner.state.phase)
    return length(getdataloader(phase, learner))
end


function starttraining(learner)
    learner.runid = uuid4()
    FitBegin() |> CallbackHandler(learner)
end


function endtraining(learner)
    learner.phase = CleanupPhase()
    FitEnd() |> CallbackHandler(learner)
end


function artifactpath(learner)
    p = joinpath(pwd(), ".artifacts", string(learner.runid))
    if !isdir(p)
        mkpath(p)
    end
    return p
end
