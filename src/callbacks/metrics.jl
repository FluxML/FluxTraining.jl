"""
    Metrics(metricfns...)

Callback that tracks metrics during training.

A metric tracking `lossfn` is included by default.

## Examples

```
metrics = Metrics(accuracy)
```

"""
struct Metrics <: Callback
    metrics
end

runafter(::Metrics) = (Recorder,)
stateaccess(::Metrics) = (
    cbstate = (metricsstep = Write(), metricsepoch = Write()),
    batch = Read()
)

function Metrics(metrics)
    return Metrics(Tuple(m isa AbstractMetric ? m : Metric(m) for m in (Loss(), metrics...)))
end


abstract type AbstractMetric end


# store metrics in `cbstate` so other callbacks can access them
function on(::Init, ::Phase, metrics::Metrics, learer)
    learner.cbstate[:metricsstep] = MVHistory()
    learner.cbstate[:metricsepoch] = DefaultDict{Phase, MVHistory}(() -> MVHistory())
end



on(::EpochBegin, ::Phase, metrics::Metrics, learner) = foreach(reset!, metrics.metrics)

function on(::BatchEnd, ::Phase, metrics::Metrics, learner)
    metricsstep = learner.cbstate[:metricsstep]
    step = learner.cbstate[:history].nsteps
    for metric in metrics.metrics
        step!(metric, learner)
        push!(metricsstep, string(metric), step, stepvalue(metric))
    end
end

function on(::EpochEnd, phase, metrics::Metrics, learner)
    metricsepoch = learner.cbstate[:metricsepoch]
    epoch = learner.cbstate[:history].epochs
    for metric in metrics.metrics
        push!(metricsepoch[phase], string(metric), step, epochvalue(metric))
    end
end


# AbstractMetric interface


mutable struct Metric{T}
    metricfn
    statistic::OnlineStats{T}
    _statistic
    name
    device
    last::T
end

function Metric(
        metricfn;
        name = uppercasefirst(string(metricfn)),
        statistic = Mean(),
        device = cpu)

    return Metric(
        metricfn,
        deepcopy(statistic),
        statistic,
        name,
        device
    )
end


function reset!(metric::Metric)
    metric.statistics = deepcopy(metric._statistics)
end

function step!(metric::Metric)
    ŷs, ys = learner.batch.ŷs, learner.batch.ys
    metric.last = metric.metricfn(ŷs, ys)
    OnlineStats.fit!(statistic, metric.last)
end

stepvalue(metric::Metric) = metric.last
epochvalue(metric::Metric) = OnlineStats.value(metric.statistic)


# Loss Metric

mutable struct Loss <: AbstractMetric
    sum
    last
    count
end

function reset!(metric::Loss)
    metric.last = nothing
    metric.sum = 0.
    metric.count = 0
end

function step!(metric::Loss, learner)
    metric.sum += learner.batch.loss
    metric.count += 1
end

stepvalue(metric::Loss) = metric.last
epochvalue(metric::Loss) = metric.sum / metric.count
