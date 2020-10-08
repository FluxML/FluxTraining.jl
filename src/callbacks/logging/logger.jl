
import .Loggables

Loggables.Image

abstract type LoggerBackend end

"""
    canlog(backend::LoggerBackend)

Return a list of [`Loggable`] types that `backend` supports.
"""
function canlog end


"""
    log_to(backend, loggable, name, i; [group])
    log_to(backends, loggable, name, i; [group])

Log `loggable` to `backend` with `name` to index `i`
of optional `group`.
"""
function log_to(backends::Tuple, loggable, name, i; group = ())
    for backend in backends
        log_to(backend, loggable, name, i, group = group)
    end
end


"""
    Logger(backends...) <: Callback
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
                group = ("Epoch", string(phase), "Metrics"))
        end
    end
end
