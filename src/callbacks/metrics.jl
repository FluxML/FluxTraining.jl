
value(metric::T) where T<:AbstractMetric = throw(
    "`value` not implemented for $(T), add a definition")

Base.show(io::IO, metric::T) where T<:AbstractMetric = print(io, string(T))

# AverageLoss
mutable struct AverageLoss <: AbstractMetric
    loss
    count
    AverageLoss() = new(nothing, nothing)
end

function on(::EpochBegin, ::AbstractFittingPhase, metric::AverageLoss, learner)
    metric.loss = 0.
    metric.count = 0
end

function on(::BatchEnd, phase::AbstractFittingPhase, metric::AverageLoss, learner)
    metric.loss += learner.state.batch.loss
    metric.count += 1
end

value(metric::AverageLoss) = metric.loss / metric.count

Base.show(io::IO, loss::AverageLoss) = print(io, "loss")

# OnlineMetrics metric

mutable struct Metric{T} <: AbstractMetric
    metricfactory
    fn
    name
    metric::OnlineStat{T}
    last::T
    device
end

"""
    Metric(fn, name = string(fn); device = cpu)

`fn(y_pred, y)`
"""
function Metric(fn, name = string(fn); device = cpu, metric = Mean)
    Metric(() -> metric(), fn, name, metric(), Inf, device)
end

Base.show(io::IO, metric::Metric) = print(io, metric.name)

value(metric::Metric) = OnlineStats.value(metric.metric)

function on(::EpochBegin, ::AbstractFittingPhase, metric::Metric, learner)
    metric.metric = metric.metricfactory()
end

function on(::BatchEnd, ::AbstractFittingPhase, metric::Metric, learner)
    metric.last = metric.fn(
            metric.device(learner.state.batch.ŷs),
            metric.device(learner.batch.batch.ys),
        )
    OnlineStats.fit!(
        metric.metric,
        metric.last,
    )
end



Accuracy() = Metric(accuracy)
