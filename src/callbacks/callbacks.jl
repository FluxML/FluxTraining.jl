
# ProgressBarLogger

mutable struct ProgressBarLogger <: AbstractCallback
    p::Union{Nothing,Progress}
end
ProgressBarLogger() = ProgressBarLogger(nothing)

function on(::EpochBegin,
        phase::AbstractFittingPhase,
        cb::ProgressBarLogger,
        learner)
    e = learner.state.history.epochs + 1
    cb.p = Progress(numsteps(learner, phase), "Epoch $(e) $(phase): ")
end

on(::BatchEnd, ::AbstractFittingPhase, cb::ProgressBarLogger, learner) = next!(cb.p)


# PrintMetrics

struct PrintMetrics <: AbstractCallback end

function on(::EpochEnd,
        phase::AbstractFittingPhase,
        cb::PrintMetrics,
        learner)
    cbs = learner.state.callbacks
    for metric in [cbs.loss, cbs.metrics...]
        println(string(metric), ": ", value(metric))
    end
end


# StopOnNaNLoss

struct StopOnNaNLoss <: AbstractCallback end

function on(::BackwardEnd, ::AbstractTrainingPhase, ::StopOnNaNLoss, learner)
    !isnan(learner.state.batch.loss) || throw(CancelFittingException("Encountered NaN loss"))
end


# Early stopping
mutable struct EarlyStopping <: AbstractCallback
    patience::Int
    waited::Int
    lowest::Float64
end
EarlyStopping(patience) = EarlyStopping(patience, 0, Inf64)

function on(::EpochEnd, ::ValidationPhase, cb::EarlyStopping, learner)
    valloss = value(learner.state.callbacks.loss)
    if (valloss > cb.lowest)
        if !(cb.waited < cb.patience)
            throw(CancelFittingException("Validation loss did not improve for $(cb.patience) epochs"))
        else
            cb.waited += 1
        end
    else
        cb.waited = 0
        cb.lowest = valloss
    end
end


struct ToGPU <: AbstractCallback end

function on(::EpochBegin, ::AbstractFittingPhase, ::ToGPU, learner)
    learner.model = gpu(learner.model)
end

function on(::BatchBegin, ::AbstractFittingPhase, cb::ToGPU, learner)
    #learner.model = gpu(learner.model)
    learner.state.batch.xs = gpu(learner.state.batch.xs)
    learner.state.batch.ys = gpu(learner.state.batch.ys)
end


garbagecollect() = (GC.gc(); ccall(:malloc_trim, Cvoid, (Cint,), 0))

"""
    GarbageCollect(nsteps)

Every `nsteps` steps, forces garbage collection.
Use this if you get memory leaks from, for example, parallel data loading.
"""
function GarbageCollect(nsteps::Int = 100)
    return CustomCallback{BatchEnd, AbstractFittingPhase}(nsteps) do learner
        garbagecollect()
    end
end
