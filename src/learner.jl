using Flux: gpu, params
using Zygote: Grads, Params
using UUIDs: UUID, uuid4


mutable struct BatchState
    batch::Union{Nothing, Tuple}
    y_pred::Union{Nothing, AbstractArray}
    loss::Union{Nothing, Real}
    gradients::Union{Nothing, Grads}
end
BatchState() = BatchState(nothing, nothing, nothing, nothing)

"""
    Learner

Central object for training that holds all necessary state.

# Fields
- `model`: a Flux model
- `databunch::`[`Databunch`](@ref): collection of `DataLoader`s
- `opt`: optimizer to update model parameters
- `lossfn`: compute losses with `lossfn(model(x), y)`
- `device`: device to train on, usually `Flux.cpu` or `Flux.gpu`
- `params`: model parameters, i.e. `Flux.params(model)`
- `phase::`[`AbstractFittingPhase`](@ref): current fitting phase
- `metrics::AbstractVector{`[`AbstractMetric`](@ref)`}`: metric callbacks to
    evaluate performance during training
- `callbacks::AbstractVector{`[`AbstractCallback`](@ref)`}`: callbacks for training loop
- `recorder::`[`Recorder`](@ref): special callback that records metrics and hyperparameters
- `scheduler::`[`Scheduler`](@ref): special callback that records metrics and hyperparameters
- `batch::`[`BatchState`](@ref): Holds training state during batch
"""
mutable struct Learner
    model
    databunch::DataBunch
    opt
    lossfn
    device
    params::Union{Params, NTuple{N, Params} where N}
    phase::AbstractFittingPhase
    metrics::AbstractVector{<:AbstractMetric}
    callbacks::AbstractVector{<:AbstractCallback}
    recorder::Recorder
    scheduler::ParamScheduler
    batch::BatchState
    runid::UUID
end

function Learner(
    model,
    databunch::DataBunch,
    opt,
    lossfn;
    device = gpu,
    metrics::AbstractVector{<:AbstractMetric} = AbstractMetric[],
    callbacks = [],
    schedule::Dict = Dict(),
    scheduler::ParamScheduler = ParamScheduler(schedule),
    use_default_metrics = true,
    use_default_callbacks = true,
    )
    if use_default_metrics
        metrics::Vector{<:AbstractMetric} = [get_default_metrics()..., metrics...]
    end
    if use_default_callbacks
        callbacks = [get_default_callbacks()..., callbacks...]
    end

    return Learner(
        model, databunch, opt, lossfn, device, params(model), InitializationPhase(),
        metrics, callbacks, Recorder(), scheduler, BatchState(), uuid4())
end

get_default_callbacks()::Vector{AbstractCallback} = [StopOnNaNLoss()]
get_default_metrics()::Vector{AbstractMetric} = [AverageLoss()]


CallbackHandler(learner::Learner) = CallbackHandler(
    learner,
    [
        learner.scheduler,
        learner.metrics...,
        learner.recorder,
        learner.callbacks...
    ])


function setschedule!(learner, schedule)
    learner.scheduler = ParamScheduler(
        delayschedule(schedule, learner.recorder.epoch)
    )
end


numsteps(learner::Learner, phase::AbstractFittingPhase) = length(
    getdataloader(learner.databunch, phase))


function starttraining(learner)
    learner.runid = uuid4()
    Initialize() |> CallbackHandler(learner)
end


function endtraining(learner)
    learner.phase = CleanupPhase()
    Cleanup() |> CallbackHandler(learner)
end


function artifactpath(learner)
    p = joinpath(pwd(), ".artifacts", string(learner.runid))
    if !isdir(p)
        mkpath(p)
    end
    return p
end
