
import .Loggables


"""
    abstract type LoggerBackend

Backend for the [`Logger`](#) callback.

To add support for logging [`Loggables.Loggable`](#) `L` to backend `B`, implement

[`log_to`](#)`(backend::B, loggable::L, names, i)`

See also [`Logger`](#), [`log_to`](#)
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
    Logger(backends...) <: Callback

Callback that logs training data to one or many [`LoggerBackend`](#)s.

Logs metrics data if `cbstate.metrics` exists (i.e. [`Metrics`](#) is used)
and hyperparameters if `cbstate.hyperparams` exists (i.e. [`Scheduler`](#) is used).

Also publishes its backends to `cbstate.loggerbackends` so that other callbacks
may use them to publish other data like images.

See also [`LoggerBackend`](#), [`Loggables.Loggable`](#), [`log_to`](#), [`TensorBoardBackend`](#)

Example:

```julia
logger = Logger(TensorBoardBackend("tblogs"))
Learner(model, data, opt, lossfn, Metrics(accuracy), logger)
```
"""
struct Logger <: Callback
    backends::Tuple
    Logger(backends...) = new(backends)
end

Base.show(io::IO, logger::Logger) = print(io, "Logger(", join(string.(logger.backends), ", "), ")")


runafter(::Logger) = (Metrics, Scheduler)
stateaccess(::Logger) = (cbstate = (
        loggerbackends = Write(),
        history = Read(),
        metricsepoch = Read(),
        metricsstep = Read(),
        hyperparams = Read(),
    ),)


function on(::Init, phase, logger::Logger, learner)
    learner.cbstate.loggerbackends = logger.backends
end


function on(::BatchEnd, phase, logger::Logger, learner)
    history = learner.cbstate.history

    # log metrics
    if haskey(learner.cbstate, :metricsstep)
        metricsstep = learner.cbstate.metricsstep
        for metric in keys(metricsstep)
            _, val = last(metricsstep, metric)
            log_to(
                logger.backends,
                Loggables.Value(val),
                string(metric),
                history.steps,
                group = ("Step", "Metrics"))
        end
    end

    # log hyperparameters
    if haskey(learner.cbstate, :hyperparams)
        hyperparams = learner.cbstate.hyperparams
        for hparam in keys(hyperparams)
            _, val = last(hyperparams, hparam)
            log_to(
                logger.backends,
                Loggables.Value(val),
                string(hparam),
                history.steps,
                group = ("Step", "HParams"))
        end
    end
end


function on(::EpochEnd, phase, logger::Logger, learner)
    history = learner.cbstate.history

    if haskey(learner.cbstate, :metricsepoch)
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
end
