
import .Loggables


"""
    abstract type LoggerBackend

Backend for logging callbacks like.

To add support for logging [`Loggables.Loggable`](#) `L` to backend `B`, implement

[`log_to`](#)`(backend::B, loggable::L, names, i)`

See also [`LogMetrics`](#), [`LogHyperParams`](#), [`log_to`](#)
"""
abstract type LoggerBackend end


"""
    log_to(backend, loggable, group, i)
    log_to(backends, loggable, group, i)

Log `loggable` to `backend` with `group` to index `i`.

- `loggable` is any [`Loggables.Loggable`](#)
- `group` can be a `String` or a tuple of `String`s implying
  some grouping which can be used by a supporting backend.
- `i` is a step counter and unique for every group.
"""
function log_to(backends::Tuple, loggable, name, i; group = ())
    for backend in backends
        log_to(backend, loggable, name, i, group = group)
    end
end


"""
    LogMetrics(backends...) <: Callback

Callback that logs step and epoch metrics to one or more [`LoggerBackend`](#)s.

See also [`LoggerBackend`](#), [`Loggables.Loggable`](#), [`log_to`](#),
[`TensorBoardBackend`](#)

Example:

```julia
logcb = LogMetrics(TensorBoardBackend("tblogs"))
Learner(model, data, opt, lossfn, Metrics(accuracy), logcb)
```
"""
struct LogMetrics <: Callback
    backends::Tuple
    LogMetrics(backends...) = new(backends)
end

stateaccess(::LogMetrics) = (
    cbstate = (history = Read(), metricsstep = Read(), metricsepoch = Read()),)


function on(::BatchEnd, phase, logger::LogMetrics, learner)
    history = learner.cbstate.history[phase]
    metricsstep = learner.cbstate.metricsstep[phase]
    for metric in keys(metricsstep)
        val = last(last(metricsstep, metric))
        log_to(
            logger.backends,
            Loggables.Value(val),
            string(metric),
            history.steps,
            group = ("Step", string(typeof(phase)), "Metrics"))
    end
end


function on(::EpochEnd, phase, logger::LogMetrics, learner)
    history = learner.cbstate.history[phase]
    metricsepoch = learner.cbstate.metricsepoch[phase]
    for metric in keys(metricsepoch)
        _, val = last(metricsepoch, metric)
        log_to(
            logger.backends,
            Loggables.Value(val),
            string(metric),
            history.epochs,
            group = ("Epoch", string(typeof(phase)), "Metrics"))
    end
end



"""
    LogHyperParams(backends...) <: Callback

Callback that logs hyperparameters to one or more [`LoggerBackend`](#)s.

See also [`LoggerBackend`](#), [`Loggables.Loggable`](#), [`log_to`](#),
[`TensorBoardBackend`](#)

## Example

```julia
logcb = LogHyperParams(TensorBoardBackend("tblogs"))
schedule = ...
Learner(model, data, opt, lossfn, Scheduler(LearningRate => schedule), logcb)
```
"""
struct LogHyperParams <: Callback
    backends::Tuple
    LogHyperParams(backends...) = new(backends)
end

stateaccess(::LogHyperParams) = (
    cbstate = (history = Read(), hyperparams = Read()),)

function on(::BatchEnd, phase, logger::LogHyperParams, learner)
    history = learner.cbstate.history[phase]
    hyperparams = learner.cbstate.hyperparams
    for hparam in keys(hyperparams)
        val = last(last(hyperparams, hparam))
        log_to(
            logger.backends,
            Loggables.Value(val),
            string(hparam),
            history.steps,
            group = ("Step", "HParams"))
    end
end


# TODO: add support for logging histograms of layer activations and gradients
"""
    LogHistograms(backends...[; freq = 100]) <: Callback

Callback that logs histograms of model weights to [`LoggerBackend`](#)s
`backends` every `freq` steps.

If histograms should be logged every step, pass `freq = nothing`
"""
struct LogHistograms <: Callback
    backends::Tuple
    function LogHistograms(backends...; freq = 100)
        if isnothing(freq)
            return new(backends)
        else
            return throttle(new(backends), BatchEnd, freq = freq)
        end
    end
end


stateaccess(::LogHistograms) = (model = Read(), cbstate = (history = Read(),))


function on(::BatchEnd, phase::AbstractTrainingPhase, logger::LogHistograms, learner)
    history = learner.cbstate.history[phase]
    log_parameters(
        logger.backends,
        learner.model,
        "Model",
        history.steps,
        group = ("Step", "Histograms", "Weights"))
end


function log_parameters(backends, x, name, epochs; group)
    params = Flux.trainable(x)
    if isempty(params) && x isa AbstractArray
        log_to(
            backends,
            Loggables.Histogram(vec(x)),
            name,
            epochs,
            group = group)
    else
        for (pname, pval) in pairs(params)
            log_parameters(
                backends,
                pval,
                "$name.$pname",
                epochs,
                group = group)
        end
    end
end


"""
    LogVisualization(visfn, backends...[; freq = 100])

Logs images created by `visfn(learner.batch)` to `backends` every `freq` steps.

See also [`BatchState`](#).
"""
struct LogVisualization <: Callback
    visfn
    backends::Tuple
    function LogVisualization(visfn, backends...; freq = 100)
        cb = new(visfn, backends)
        if isnothing(freq)
            return cb
        else
            return throttle(cb, BatchEnd, freq = freq)
        end
    end

end


stateaccess(::LogVisualization) = (batch = Read(), cbstate = (history = Read(),))

function on(::BatchEnd, phase::AbstractTrainingPhase, logger::LogVisualization, learner)
    history = learner.cbstate.history[phase]
    image = logger.visfn(learner.batch)

    log_to(
        logger.backends,
        Loggables.Image(image),
        "Visualization",
        history.steps,
        group = ("Step",))
end
