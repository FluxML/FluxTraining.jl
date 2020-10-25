
# ProgressPrinter

"""
    ProgressPrinter()

Prints a progress bar of the currently running epoch.
"""
mutable struct ProgressPrinter <: Callback
    p::Union{Nothing,Progress}
end
ProgressPrinter() = ProgressPrinter(nothing)
Base.show(io::IO, ::ProgressPrinter) = print(io, "ProgressPrinter()")

function on(::EpochBegin,
        phase::Phase,
        cb::ProgressPrinter,
        learner)
    e = learner.cbstate.history.epochs + 1
    cb.p = Progress(numsteps(learner, phase), "Epoch $(e) $(phase): ")
end

on(::BatchEnd, ::Phase, cb::ProgressPrinter, learner) = next!(cb.p)

runafter(::ProgressPrinter) = (Recorder,)
stateaccess(::ProgressPrinter) = (data = Read(), cbstate = (history = Read()),)


"""
    MetricsPrinter()

Prints any metrics after every epoch.
"""
struct MetricsPrinter <: Callback end

function on(::EpochEnd,
        phase::Phase,
        cb::MetricsPrinter,
        learner)
    mvhistory = learner.cbstate.metricsepoch[phase]
    for key in keys(mvhistory)
        println(key, ": ", last(mvhistory, key)[2])
    end
end

stateaccess(::MetricsPrinter) = (; cbstate = (; metricsepoch = Read()))
runafter(::MetricsPrinter) = (Metrics,)

# StopOnNaNLoss

"""
    StopOnNaNLoss()

Stops the training when a NaN loss is encountered.
"""
struct StopOnNaNLoss <: SafeCallback end

function on(::BackwardEnd, ::AbstractTrainingPhase, ::StopOnNaNLoss, learner)
    !isnan(learner.batch.loss) || throw(CancelFittingException("Encountered NaN loss"))
end

stateaccess(::StopOnNaNLoss) = (batch = (loss = Read()),)


# Early stopping
mutable struct EarlyStopping <: SafeCallback
    patience::Int
    waited::Int
    lowest::Float64
end
EarlyStopping(patience) = EarlyStopping(patience, 0, Inf64)

function on(::EpochEnd, ::ValidationPhase, cb::EarlyStopping, learner)
    valloss = last(learner.cbstate.metricsepoch[phase], :Loss)[2]
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

stateaccess(::EarlyStopping) = (callbacks = Read(), cbstate = (; metricsepoch = Read()))


struct ToGPU <: SafeCallback end

function on(::EpochBegin, ::Phase, ::ToGPU, learner)
    model!(learner, gpu(learner.model))
end

stateaccess(::ToGPU) = (model = Write(), params = Write(), batch = (xs = Write(), ys = Write()),)

function on(::BatchBegin, ::Phase, cb::ToGPU, learner)
    learner.batch.xs = gpu(learner.batch.xs)
    learner.batch.ys = gpu(learner.batch.ys)
end


garbagecollect() = (GC.gc(); ccall(:malloc_trim, Cvoid, (Cint,), 0))

"""
    GarbageCollect(nsteps)

Every `nsteps` steps, forces garbage collection.
Use this if you get memory leaks from, for example, parallel data loading.
"""
function GarbageCollect(nsteps::Int = 100)
    return throttle(CustomCallback((learner) -> garbagecollect(), BatchEnd), freq = nsteps)
end
