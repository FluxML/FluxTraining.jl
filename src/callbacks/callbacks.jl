
# ProgressBarLogger

mutable struct ProgressBarLogger <: AbstractCallback
    p::Union{Nothing,Progress}
end
ProgressBarLogger() = ProgressBarLogger(nothing)

function on(::EpochBegin,
        phase::AbstractFittingPhase,
        cb::ProgressBarLogger,
        learner)
    e = learner.recorder.epoch
    cb.p = Progress(numsteps(learner, phase), "Epoch $(e) $(phase): ")
end

on(::BatchEnd, ::AbstractFittingPhase, cb::ProgressBarLogger, learner) = next!(cb.p)


# PrintMetrics

struct PrintMetrics <: AbstractCallback end

function on(::EpochEnd,
        phase::AbstractFittingPhase,
        cb::PrintMetrics,
        learner)
    for metric in learner.metrics
        println(string(metric), ": ", value(metric))
    end
end


# StopOnNaNLoss

struct StopOnNaNLoss <: AbstractCallback end

struct NaNLossException <: Exception end

function on(::BackwardEnd, ::AbstractTrainingPhase, ::StopOnNaNLoss, learner)
    !isnan(learner.batch.loss) || throw(CancelFittingException("Encountered NaN loss"))
end


# Early stopping
mutable struct EarlyStopping <: AbstractCallback
    patience::Int
    waited::Int
    lowest::Float64
end
EarlyStopping(patience) = EarlyStopping(patience, 0, Inf64)

function on(::EpochEnd, ::ValidationPhase, cb::EarlyStopping, learner)
    valloss = value(learner.metrics[1])
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



garbagecollect() = (GC.gc(); ccall(:malloc_trim, Cvoid, (Cint,), 0))

"""
    GarbageCollect(nsteps)

Every `nsteps`, forces garbage collection.
Use this if you get memory leaks from, for example, parallel data loading.
"""
function GarbageCollect(nsteps::Int = 100)
    return CustomCallback{BatchEnd, AbstractFittingPhase}(nsteps) do learner
        garbagecollect()
    end
end
