
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

mutable struct Metric{T} <: AbstractMetric
    metricfactory
    fn
    name
    metric::OnlineStat{T}
    last::T
end
Metric(metricfactory, fn, name = string(fn)) = Metric(metricfactory, fn, name, metricfactory(), Inf)

Base.show(io::IO, metric::Metric) = print(io, metric.name)

value(metric::Metric) = OnlineStats.value(metric.metric)

function on(::EpochBegin, ::AbstractFittingPhase, metric::Metric, learner)
    metric.metric = metric.metricfactory()
end

function on(::BatchEnd, phase::AbstractFittingPhase, metric::Metric, learner)
    metric.last = metric.fn(
            # FIXME: configure which device to evaluate on
            learner.batch.y_pred,# |> cpu,
            learner.batch.batch[2],# |> cpu
        )
    OnlineStats.fit!(
        metric.metric,
        metric.last,
    )
end

MeanMetric(fn, name = string(fn); weight = EqualWeight()) =
    Metric(() -> Mean(Float64, weight = weight), fn, name)


Accuracy() = MeanMetric(accuracy)
