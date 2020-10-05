
# store metrics in `cbstate` so other callbacks can access them
function on(::Init, ::Phase, metric::AbstractMetric, learer)
    metricsstep = get(learner.cbstate, :metricsstep)
    if isnothing(metricsstep)
        learner.cbstate[:metricsstep] = MVHistory()
    end

    metricsepoch = get(learner.cbstate, :metricsstep)
    if isnothing(metricsstep)
        learner.cbstate[:metricsepoch] = MVHistory()
    end
end

# Loss
mutable struct Loss <: AbstractMetric
    loss
    last
    count
    Loss() = new(nothing, nothing, nothing)
end


function on(::EpochBegin, ::Phase, metric::Loss, learner)
    metric.loss = 0.
    metric.count = 0
end

function on(::BatchEnd, phase::Phase, metric::Loss, learner)
    metric.loss += learner.batch.loss
    metric.last = learner.batch.loss
    metric.count += 1
end

stateaccess(::Loss) = (
    batch = (loss = Read(),),
    cbstate = (metricsstep = Write(), metricsepoch = Write()))

resolveconflict(::AbstractMetric, ::AbstractMetric) = NoConflict()

stepvalue(metric::Loss) = metric.last
epochvalue(metric::Loss) = metric.loss / metric.count

Base.show(io::IO, loss::Loss) = print(io, "Loss()")

# OnlineMetrics metric

mutable struct Metric{T} <: AbstractMetric
    metricfactory
    fn
    name
    metric::OnlineStat{T}
    last::T
    device
end

stepvalue(metric::Metric) = metric.last
epochvalue(metric::Metric) = OnlineStats.value(metric.metric)

"""
    Metric(fn, name = string(fn); device = cpu)

`fn(y_pred, y)`
"""
function Metric(fn, name = string(fn); device = cpu, metric = Mean)
    Metric(() -> metric(), fn, name, metric(), Inf, device)
end

Base.show(io::IO, metric::Metric) = print(io, "Metric(", metric.name, ")")


function on(::EpochBegin, ::Phase, metric::Metric, learner)
    metric.metric = metric.metricfactory()
end

function on(::BatchEnd, ::Phase, metric::Metric, learner)
    metric.last = metric.fn(
            metric.device(learner.batch.ŷs),
            metric.device(learner.batch.ys),
        )
    OnlineStats.fit!(
        metric.metric,
        metric.last,
    )
end

stateaccess(::Metric) = (batch = (ŷs = Read(), ys = Read()),)


Accuracy() = Metric(accuracy)
