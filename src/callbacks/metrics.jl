import Flux: cpu
import OnlineStats
using OnlineStats: EqualWeight, Mean, OnlineStat


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
    metric.loss += learner.batch.loss
    metric.count += 1
end

value(metric::AverageLoss) = metric.loss / metric.count

Base.show(io::IO, loss::AverageLoss) = print(io, "loss")

# OnlineMetrics metric

mutable struct Metric <: AbstractMetric
    metricfactory
    fn
    metric::OnlineStat
end
Metric(metricfactory, fn) = Metric(metricfactory, fn, metricfactory())

Base.show(io::IO, metric::Metric) = print(io, string(metric.fn))

value(metric::Metric) = OnlineStats.value(metric.metric)

function on(::EpochBegin, ::AbstractFittingPhase, metric::Metric, learner)
    metric.metric = metric.metricfactory()
end

function on(::BatchEnd, phase::AbstractFittingPhase, metric::Metric, learner)
    OnlineStats.fit!(
        metric.metric,
        metric.fn(
            learner.batch.y_pred |> cpu,
            learner.batch.batch[2] |> cpu
        )
    )
end

MeanMetric(fn, weight = EqualWeight()) = Metric(() -> Mean(Float32, weight = weight), fn)


Accuracy() = MeanMetric(accuracy)