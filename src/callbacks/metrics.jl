"""
    Metrics(metrics...)

Callback that tracks metrics during training.

`metrics` can be both [`AbstractMetric`](#)s or functions
like `f(ŷs, ys)` which will be converted to [`Metric`](#)s.

A metric tracking `lossfn` is included by default.

## Examples

`metrics = Metrics(accuracy)`
`metrics = Metrics(Metric(Flux.mse, device = gpu), Metric(Flux.mae, device = gpu))`

"""
struct Metrics <: Callback
    metrics::Tuple
    function Metrics(metrics...)
        return new(Tuple(m isa AbstractMetric ? m : Metric(m) for m in (Loss(), metrics...)))
    end
end

runafter(::Metrics) = (Recorder,)
stateaccess(::Metrics) = (
    cbstate = (metricsstep = Write(), metricsepoch = Write(), history = Read()),
    batch = Read()
)

Base.show(io::IO, metrics::Metrics) = print(io, "Metrics(", join(string.(metrics.metrics), ", "), ")")


# store metrics in `cbstate` so other callbacks can access them
function on(::Init, ::Phase, metrics::Metrics, learner)
    learner.cbstate.metricsstep = MVHistory()
    learner.cbstate.metricsepoch = DefaultDict{Phase, MVHistory}(() -> MVHistory())
end



on(::EpochBegin, ::Phase, metrics::Metrics, learner) = foreach(reset!, metrics.metrics)

function on(::BatchEnd, phase, metrics::Metrics, learner)
    metricsstep = learner.cbstate.metricsstep
    step = learner.cbstate.history.steps
    for metric in metrics.metrics
        step!(metric, learner)
        if phase isa AbstractTrainingPhase
            push!(metricsstep, Symbol(metricname(metric)), step, stepvalue(metric))
        end
    end
end

function on(::EpochEnd, phase, metrics::Metrics, learner)
    metricsepoch = learner.cbstate.metricsepoch
    epoch = learner.cbstate.history.epochs
    for metric in metrics.metrics
        push!(metricsepoch[phase], Symbol(metricname(metric)), epoch, epochvalue(metric))
    end
end


# AbstractMetric interface


"""
    abstract type AbstractMetric

Abstract type for metrics passed to [`Metrics`](#).

For most use cases, you should use [`Metric`](#), the standard
implementation.

## Interface

If [`Metric`](#) doesn't fit your use case, you can create
a new subtype of `AbstractMetric` and implement the following
methods to make it compatible with [`Metrics`](#):

- [`reset!`](#)`(metric)`
- [`step!`](#)`(metric, learner)`
- [`stepvalue`](#)`(metric)`
- [`epochvalue`](#)`(metric)`
- [`metricname`](#)`(metric)`
"""
abstract type AbstractMetric end

mutable struct Metric{T}
    metricfn
    statistic::OnlineStat{T}
    _statistic
    name
    device
    last::Union{Nothing, T}
end

Base.show(io::IO, metric::Metric{T}) where T = print(io, "Metric(", metric.name, ")")

"""
    Metric(metricfn[; statistic, device, name])

Implementation of [`AbstractMetric`](#) that can be used with the
[`Metrics`](#) callback.

## Arguments

- `metricfn(ŷs, ys)` should return a number.
- `statistic` is a `OnlineStats.Statistic` that is updated after every step.
    The default is `OnlineStats.Mean()`
- `name` is used for printing.
- `device = cpu` is a function applied to `ys` and `ŷs` before calling
    `metricfn`. If `metricfn` works on the GPU, you may want to pass
    `Flux.gpu` here for better performance.

## Examples

- `Metric(accuracy)`
- `Metric(Flux.mse, device = gpu)`
- `Metric(Flux.mae, device = gpu)`

"""
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
        device,
        nothing,
    )
end


function reset!(metric::Metric)
    metric.statistic = deepcopy(metric._statistic)
end

function step!(metric::Metric, learner)
    ŷs, ys = learner.batch.ŷs, learner.batch.ys
    metric.last = metric.metricfn(ŷs, ys)
    OnlineStats.fit!(metric.statistic, metric.last)
end

stepvalue(metric::Metric) = metric.last
epochvalue(metric::Metric) = OnlineStats.value(metric.statistic)
metricname(metric::Metric) = metric.name


# Loss Metric

mutable struct Loss <: AbstractMetric
    sum
    last
    count
end

Loss() = Loss(0, Inf, 0)
Base.show(io::IO, loss::Loss) = print(io, "Loss()")

function reset!(metric::Loss)
    metric.last = nothing
    metric.sum = 0.
    metric.count = 0
end

function step!(metric::Loss, learner)
    metric.last = learner.batch.loss
    metric.sum += learner.batch.loss
    metric.count += 1
end

stepvalue(metric::Loss) = metric.last
epochvalue(metric::Loss) = metric.sum / metric.count
metricname(metric::Loss) = "Loss"
