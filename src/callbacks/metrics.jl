"""
    Metrics(metrics...) <: Callback

Callback that tracks metrics during training.

You can pass any number of `metrics` with every argument being
- an [`AbstractMetric`](#) like [`Metric`](#); or
- a function `f(ŷs, ys) -> val`

This callback is added by default to every [`Learner`](#) unless you pass in
`usedefaultcallbacks = false`. A metric tracking `learner.lossfn` [`Loss`](#)
is included by default.

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
        return new(
            Tuple(m isa AbstractMetric ? m : Metric(m) for m in (Loss(), metrics...)),
        )
    end
end

runafter(::Metrics) = (Recorder,)
stateaccess(::Metrics) = (
    cbstate = (metricsstep = Write(), metricsepoch = Write(), history = Read()),
    step = Read(),
)

Base.show(io::IO, metrics::Metrics) =
    print(io, "Metrics(", join(string.(metrics.metrics), ", "), ")")


# store metrics in `cbstate` so other callbacks can access them
function init!(metrics::Metrics, learner)
    length(metrics.metrics) == length(unique(metricname.(metrics.metrics))) ||
        error("Multiple metrics have the same name!")
    if !haskey(learner.cbstate, :metricsstep)
        learner.cbstate.metricsstep = DefaultDict{Phase,MVHistory}(() -> MVHistory())
    end
    if !haskey(learner.cbstate, :metricsepoch)
        learner.cbstate.metricsepoch = DefaultDict{Phase,MVHistory}(() -> MVHistory())
    end
end



on(::EpochBegin, ::Phase, metrics::Metrics, learner) = foreach(reset!, metrics.metrics)

function on(::StepEnd, phase, metrics::Metrics, learner)
    metricsstep = learner.cbstate.metricsstep[phase]
    step = learner.cbstate.history[phase].steps
    for metric in metrics.metrics
        step!(metric, learner, phase)
        val = stepvalue(metric)
        if val !== nothing
            metricsstep = learner.cbstate.metricsstep[phase]
            push!(metricsstep, Symbol(metricname(metric)), step, val)
        end
    end
end

function on(::EpochEnd, phase, metrics::Metrics, learner)
    epoch = learner.cbstate.history[phase].epochs
    for metric in metrics.metrics
        val = epochvalue(metric)
        if val !== nothing
            metricsepoch = learner.cbstate.metricsepoch[phase]
            push!(metricsepoch, Symbol(metricname(metric)), epoch, val)
        end
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

step!(metric, learner, _) = step!(metric, learner)

mutable struct Metric{T} <: AbstractMetric
    metricfn::Any
    statistic::OnlineStat{T}
    _statistic::Any
    name::Any
    device::Any
    P::Any
    last::Union{Nothing,T}
end

Base.show(io::IO, metric::Metric{T}) where {T} = print(io, "Metric(", metric.name, ")")

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
- `phase = Phase`: a (sub)type of [`Phase`](#) that restricts for which phases the
    metric is computed.

## Examples

- `Metric(accuracy)`
- `Metric(Flux.mse, device = gpu, name = "Mean Squared Error")`
- `Metric(Flux.mae, device = gpu)`

```julia
cb = Metric(Flux.mse, device = gpu, name = "Mean Squared Error")
```

If a metric is expensive to compute and you don't want it to slow down the
training phase, you can compute it on the validation phase only:

```julia
cb = Metric(expensivemetric, P = ValidationPhase)
```
"""
function Metric(
    metricfn;
    name = uppercasefirst(string(metricfn)),
    statistic = Mean(),
    device = cpu,
    phase = Phase,
)

    return Metric(metricfn, deepcopy(statistic), statistic, name, device, phase, nothing)
end


function reset!(metric::Metric)
    metric.statistic = deepcopy(metric._statistic)
end

function step!(metric::Metric, learner, phase)
    if phase isa metric.P
        ŷs, ys = metric.device((learner.step.ŷs, learner.step.ys))
        metric.last = metric.metricfn(ŷs, ys)
        OnlineStats.fit!(metric.statistic, metric.last)
    else
        metric.last = nothing
    end
end

stepvalue(metric::Metric) = metric.last
function epochvalue(metric::Metric)
    if isnothing(metric.last)
        nothing
    else
        OnlineStats.value(metric.statistic)
    end
end
metricname(metric::Metric) = metric.name


# Loss Metric

mutable struct Loss <: AbstractMetric
    statistic::Any
    _statistic::Any
    last::Any
    name::Any
end


function Loss(weight = EqualWeight(); name = "Loss")
    stat = Mean(weight = weight)
    return Loss(deepcopy(stat), stat, nothing, name)
end


function reset!(loss::Loss)
    loss.statistic = deepcopy(loss._statistic)
end


function step!(metric::Loss, learner, _)
    metric.last = learner.step.loss
    OnlineStats.fit!(metric.statistic, metric.last)
end


Base.show(io::IO, ::Loss) = print(io, "Loss()")


stepvalue(metric::Loss) = metric.last
epochvalue(metric::Loss) = OnlineStats.value(metric.statistic)
metricname(metric::Loss) = metric.name


# Utility


function SmoothLoss(β = 0.02)
    return Loss(OnlineStats.ExponentialWeight(1 - β), name = "SmoothLoss")
end

@testset "Metric" begin
    cb = Metrics(Metric(accuracy, phase = ValidationPhase))
    learner = testlearner(Recorder(), cb)
    @test_nowarn fit!(learner, 1)
    @test :Accuracy ∈ keys(learner.cbstate.metricsstep[ValidationPhase()])
    @test !(:Accuracy ∈ keys(learner.cbstate.metricsstep[TrainingPhase()]))
    @test :Accuracy ∈ keys(learner.cbstate.metricsepoch[ValidationPhase()])
    @test !(:Accuracy ∈ keys(learner.cbstate.metricsepoch[TrainingPhase()]))
end
