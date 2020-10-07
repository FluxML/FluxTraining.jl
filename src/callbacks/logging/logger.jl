
import .Loggables

Loggables.Image

abstract type LoggerBackend end

"""
    canlog(backend::LoggerBackend)

Return a list of [`Loggable`] types that `backend` supports.
"""
function canlog end


"""
    log_to(backend, loggable, name, i, [group])

Log `loggable` to `backend` with `name` to index `i`
of optional `group`.
"""
function log_to end


"""
    Logger(backends...) <: Callback
"""
struct Logger <: Callback
    backends::Tuple
end
Logger(backends...) = Logger(backends)


runafter(::Logger) = (Metrics, Scheduler)
stateaccess(::Logger) = (cbstate = (
        loggerbackends = Write(),
        history = Read(),
        metricsepoch = Read(),
        metricsstep = Read(),
        hyperparams = Read(),
    ),)


function on(::Init, phase, logger::Logger, learner)
    learner.cbstate[:loggerbackends] = logger.backends
end


function on(::BatchEnd, phase, logger::Logger, learner)
    history = learner.cbstate.history

    metricsstep = get(learner.cbstate, :metricsstep)
    if !isnothing(metrics)
        for (metric, history) in metricsstep
            log_to(
                logger.backends,
                Value(last(history)[2]),
                string(metric),
                history.steps,
                group = "step")
        end
    end

    hyperparams = get(learner.cbstate, :hyperparams)
    if !isnothing(hyperparams)
        for h in keys(hyperparams)
            _, val = last(hyperparams, h)
            log_to(
                logger.backends,
                Value(val),
                string(h),
                history.steps,
                group = "step")
        end
    end
end


function on(::EpochEnd, phase, logger::Logger, learner)
    history = learner.cbstate.history

    metrics = get(learner.cbstate, :metrics)
    if !isnothing(metrics)
        for metric in values(metrics)
            log(
                logger.backends,
                Value(epochvalue(metric)),
                string(metric),
                history.epochs,
                group = ("epoch", string(phase)))
        end
    end
end
