using Flux: gpu


mutable struct Learner
    databunch::DataBunch
    model
    opt
    lossfn
    device
    metrics::AbstractVector{<:AbstractMetric}
    callbacks::AbstractVector{<:AbstractCallback}
    recorder::Recorder
    state::TrainingState
end

TrainingState(learner::Learner) = TrainingState(
    learner, params(learner.model), nothing, 0, 0, nothing, nothing, nothing, nothing, nothing, nothing)

function Learner(
    databunch::DataBunch,
    model,
    opt,
    lossfn;
    device = gpu,
    metrics = [],
    callbacks = [],
    use_default_metrics = true,
    use_default_callbacks = true,
    )
    if use_default_metrics
        metrics = [get_default_metrics()..., metrics...]
    end
    if use_default_callbacks
        callbacks = [get_default_callbacks()..., callbacks...]
    end

    learner = Learner(databunch, model, opt, lossfn, device, metrics, callbacks, Recorder(), TrainingState())
    learner.state = TrainingState(learner)
    return learner
end

get_default_callbacks() = [StopOnNaNLoss()]
get_default_metrics() = [AverageLoss()]
