"""
    Metrics(metrics...) <: Callback

Callback that tracks metrics during training.

You can pass any number of `metrics` with every argument being
- an [`AbstractMetric`](#)s; or
- a function `f(ŷs, ys) -> val`

A metric tracking `learner.lossfn` [`Loss`](#) is included by default. See
also [`Metric`](#).

The computed metrics can be access in `learner.cbstate.metricsstep` and
`learner.cbstate.metricsepoch` for steps and epochs, respectively.

## Examples

Track [`accuracy`](#):

```julia
cb = Metrics(accuracy)
```

Pass in [`Metric`]s:

```julia
cb = Metrics(
    Metric(Flux.mse, device = gpu),
    Metric(Flux.mae, device = gpu)
)
```

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
    step = Read()
)

Base.show(io::IO, metrics::Metrics) = print(io, "Metrics(", join(string.(metrics.metrics), ", "), ")")


# store metrics in `cbstate` so other callbacks can access them
function init!(metrics::Metrics, learner)
    length(metrics.metrics) == length(unique(metricname.(metrics.metrics))) || error("Multiple metrics have the same name!")
    if !haskey(learner.cbstate, :metricsstep)
        learner.cbstate.metricsstep = DefaultDict{Phase, MVHistory}(() -> MVHistory())
    end
    if !haskey(learner.cbstate, :metricsepoch)
        learner.cbstate.metricsepoch = DefaultDict{Phase, MVHistory}(() -> MVHistory())
    end
end



on(::EpochBegin, ::Phase, metrics::Metrics, learner) = foreach(reset!, metrics.metrics)

function on(::StepEnd, phase, metrics::Metrics, learner)
    metricsstep = learner.cbstate.metricsstep[phase]
    step = learner.cbstate.history[phase].steps
    for metric in metrics.metrics
        step!(metric, learner)
        push!(metricsstep, Symbol(metricname(metric)), step, stepvalue(metric))
    end
end

function on(::EpochEnd, phase, metrics::Metrics, learner)
    metricsepoch = learner.cbstate.metricsepoch[phase]
    epoch = learner.cbstate.history[phase].epochs
    for metric in metrics.metrics
        push!(metricsepoch, Symbol(metricname(metric)), epoch, epochvalue(metric))
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

mutable struct Metric{T} <: AbstractMetric
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

Positional:

- `metricfn(ŷs, ys)` should return a number.

Keyword:

- `statistic` is a `OnlineStats.Statistic` that is updated after every step.
    The default is `OnlineStats.Mean()`
- `name` is used for printing.
- `device` is a function applied to `ŷs` and `ys`
    before passing them to `metricfn`. The default is `Flux.cpu` so that
    the callback works if `metricfn` doesn't support arrays from other device
    types. If, for example, `metricfn` works on `CurArray`s, you can pass
    `device = Flux.gpu`.

## Examples

- `Metric(accuracy)`
- `Metric(Flux.mse, device = gpu, name = "Mean Squared Error")`
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
    ŷs, ys = metric.device((learner.step.ŷs, learner.step.ys))
    metric.last = metric.metricfn(ŷs, ys)
    OnlineStats.fit!(metric.statistic, metric.last)
end

stepvalue(metric::Metric) = metric.last
epochvalue(metric::Metric) = OnlineStats.value(metric.statistic)
metricname(metric::Metric) = metric.name


# Loss Metric

mutable struct Loss <: AbstractMetric
    statistic
    _statistic
    last
    name
end


function Loss(weight = EqualWeight(); name = "Loss")
    stat = Mean(weight = weight)
    return Loss(deepcopy(stat), stat, nothing, name)
end


function reset!(loss::Loss)
    loss.statistic = deepcopy(loss._statistic)
end


function step!(metric::Loss, learner)
    metric.last = learner.step.loss
    OnlineStats.fit!(metric.statistic, metric.last)
end


Base.show(io::IO, loss::Loss) = print(io, "Loss()")


stepvalue(metric::Loss) = metric.last
epochvalue(metric::Loss) = OnlineStats.value(metric.statistic)
metricname(metric::Loss) = metric.name


# Utility


function SmoothLoss(β = 0.02)
    return Loss(OnlineStats.ExponentialWeight(1-β), name = "SmoothLoss")
end
