import Flux: cpu
import OnlineStats
using OnlineStats: EqualWeight, Mean, OnlineStat

abstract type AbstractMetric <: AbstractCallback end

value(metric::T) where T<:AbstractMetric = throw(
    "`value` not implemented for $(T), add a definition")
order(metric::Type{<:AbstractMetric}) = -100

Base.show(io::IO, metric::T) where T<:AbstractMetric = print(io, string(T))

# AverageMetric
mutable struct AverageMetric <: AbstractMetric
    fn
    value
    count
    AverageMetric(fn) = new(fn, nothing, nothing)
end

function on_epoch_begin(metric::AverageMetric, state::TrainingState, phase::AbstractFittingPhase )
    metric.value = 0.
    metric.count = 0
end

function on_batch_end(metric::AverageMetric, state::TrainingState, phase::AbstractFittingPhase )
    metric.value += metric.fn(state.output |> cpu, state.batchy |> cpu)
    metric.count += 1
end

value(metric::AverageMetric) = metric.value / metric.count

function Base.show(io::IO, metric::AverageMetric)
    fnname = titlecase(string(metric.fn))
    print(io, fnname)
end


# AverageLoss
mutable struct AverageLoss <: AbstractMetric
    loss
    count
    AverageLoss() = new(nothing, nothing)
end

function on_epoch_begin(metric::AverageLoss, state::TrainingState, phase::AbstractFittingPhase )
    metric.loss = 0.
    metric.count = 0
end

function on_batch_end(metric::AverageLoss, state::TrainingState, phase::AbstractFittingPhase )
    metric.loss += state.lossbatch
    metric.count += length(getdataloader(state.learner.databunch, phase))
end

value(metric::AverageLoss) = metric.loss / metric.count

Base.show(io::IO, loss::AverageLoss) = print(io, "Loss")

# OnlineMetrics metric


mutable struct Metric <: AbstractMetric
    metricfn
    fn
    metric::OnlineStat
end

Metric(metricfn, fn) = Metric(metricfn, fn, metricfn())

value(metric::Metric) = OnlineStats.value(metric.metric)


MeanMetric(fn, weight = EqualWeight()) = Metric(() -> Mean(weight = weight), fn)

function on_epoch_begin(metric::Metric, state::TrainingState, phase::AbstractFittingPhase)
    metric.metric = metric.metricfn()
end

function on_batch_end(metric::Metric, state::TrainingState, phase::AbstractFittingPhase)
    OnlineStats.fit!(
        metric.metric, metric.fn(state.output |> cpu, state.batchy |> cpu))
end
