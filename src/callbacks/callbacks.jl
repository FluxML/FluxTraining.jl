
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

Prints metrics after every epoch. Relies on [`Metrics`](#).
"""
struct MetricsPrinter <: Callback end

function on(::EpochEnd,
        phase::Phase,
        cb::MetricsPrinter,
        learner)
    mvhistory = learner.cbstate.metricsepoch[phase]
    epoch = learner.cbstate.history.epochs
    print_epoch_table(mvhistory, epoch, phase)
end


function print_epoch_table(mvhistory, epoch, phase)
    header = vcat(["Phase", "Epoch"], string.(keys(mvhistory)))
    vals = [last(mvhistory, key) |> last for key in keys(mvhistory)]
    data = reshape(vcat([string(typeof(phase)), epoch], vals), 1, :)
    pretty_table(data, header, formatters = PrettyTables.ft_round(5))
end

stateaccess(::MetricsPrinter) = (; cbstate = (metricsepoch = Read(), history = Read()))
runafter(::MetricsPrinter) = (Metrics,)

# StopOnNaNLoss

"""
    StopOnNaNLoss()

Stops the training when a NaN loss is encountered.
"""
struct StopOnNaNLoss <: Callback end

function on(::BackwardEnd, ::AbstractTrainingPhase, ::StopOnNaNLoss, learner)
    !isnan(learner.batch.loss) || throw(CancelFittingException("Encountered NaN loss"))
end

stateaccess(::StopOnNaNLoss) = (batch = (loss = Read()),)


# Early stopping
"""
    EarlyStopping(patience)

Stops training if validation loss hasn't decreased in `patience` epochs.
"""
mutable struct EarlyStopping <: Callback
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


"""
    ToGPU()

Callback that moves model and batch data to the GPU during training.
"""
struct ToGPU <: Callback end

function on(::EpochBegin, ::Phase, ::ToGPU, learner)
    model!(learner, gpu(learner.model))
end

stateaccess(::ToGPU) = (
    model = Write(),
    params = Write(),
    batch = (xs = Write(), ys = Write()),
)

function on(::BatchBegin, ::Phase, cb::ToGPU, learner)
    learner.batch.xs = gpu(learner.batch.xs)
    learner.batch.ys = gpu(learner.batch.ys)
end


function garbagecollect()
    GC.gc()
    if Base.Sys.islinux()
        ccall(:malloc_trim, Cvoid, (Cint,), 0))
    end
end


"""
    GarbageCollect(nsteps)

Every `nsteps` steps, forces garbage collection.
Use this if you get memory leaks from, for example,
parallel data loading.

Performs an additional C-call on Linux systems that can
sometimes help.
"""
function GarbageCollect(nsteps::Int = 100)
    return throttle(CustomCallback((learner) -> garbagecollect(), BatchEnd), freq = nsteps)
end
