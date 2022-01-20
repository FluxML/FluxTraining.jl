
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
    e = learner.cbstate.history[phase].epochs + 1
    dataiter = get(learner.data, phasedataiter(phase), nothing)
    if isnothing(dataiter)
        cb.p = nothing
        println("Epoch $(e) $(phase) ...")
    else
        cb.p = Progress(length(dataiter), "Epoch $(e) $(phase): ")
    end
end

on(::StepEnd, ::Phase, cb::ProgressPrinter, learner) = isnothing(cb.p) || next!(cb.p)

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
    epoch = learner.cbstate.history[phase].epochs
    print_epoch_table(mvhistory, epoch, phase)
end


function print_epoch_table(mvhistory, epoch, phase)
    header = vcat(["Phase", "Epoch"], string.(keys(mvhistory)))
    vals = [last(mvhistory, key) |> last for key in keys(mvhistory)]
    data = reshape(vcat([string(typeof(phase)), epoch], vals), 1, :)
    pretty_table(data; header = header, formatters = PrettyTables.ft_round(5))
end

stateaccess(::MetricsPrinter) = (; cbstate = (metricsepoch = Read(), history = Read()))
runafter(::MetricsPrinter) = (Metrics, TimingRecorder)

# Logging Results to CSV file
struct CSVMetricsLogger <: Callback
    verbose::Bool
    basedir::String
    basename::String
    # When do I close the file handles?
    file_handles::Dict{String,IOStream}

    function CSVMetricsLogger(;basedir::String = "./", basename::String = "", verbose::Bool = false)
        if !isdir(basedir)
            throw(ArgumentError("CSVMetricsLogger: basedir '$(basedir)' does not exist"))
        end
        return new(verbose, basedir, basename, Dict{String,IOStream}())
    end
end

function get_file_handle(cb::CSVMetricsLogger, phase::Phase)
    phasename = string(typeof(phase))
    if !haskey(cb.file_handles, phasename)
        filename = (length(cb.basename) == 0 ? "" : (cb.basename * "-")) * phasename * "-" * string(now()) * ".csv"
        path = joinpath(cb.basedir, filename)
        if cb.verbose
            Base.printstyled("[CSVMetricsLogger]: $(phasename) is being logged to $(path)\n"; color = :blue, bold = true)
        end
        cb.file_handles[phasename] = open(path, "w")
        return cb.file_handles[phasename], false
    end
    return cb.file_handles[phasename], true
end

function on(::EpochEnd,
        phase::Phase,
        cb::CSVMetricsLogger,
        learner)
    iostream, header_written = get_file_handle(cb, phase)

    mvhistory = learner.cbstate.metricsepoch[phase]
    epoch = learner.cbstate.history[phase].epochs
    
    if !header_written
        header = join(vcat(["Epoch"], string.(keys(mvhistory))), ",")
        println(iostream, header)
    end

    vals = [last(mvhistory, key) |> last for key in keys(mvhistory)]
    data = join(vcat([epoch], vals), ",")
    println(iostream, data)

    flush(iostream)
    return
end

stateaccess(::CSVMetricsLogger) = (; cbstate = (metricsepoch = Read(), history = Read()))
runafter(::CSVMetricsLogger) = (Metrics, TimingRecorder)

# StopOnNaNLoss

"""
    StopOnNaNLoss()

Stops the training when a NaN loss is encountered.
"""
struct StopOnNaNLoss <: Callback end

function on(::BackwardEnd, ::AbstractTrainingPhase, ::StopOnNaNLoss, learner)
    !isnan(learner.step.loss) || throw(CancelFittingException("Encountered NaN loss"))
end

stateaccess(::StopOnNaNLoss) = (step = (loss = Read()),)


"""
    ToGPU()

Callback that moves model and batch data to the GPU during training.
"""
ToGPU() = ToDevice(gpu, gpu)
"""
    ToDevice(movefn[, movemodelfn]) <: Callback

Moves model and step data to a device using `movedatafn` for step data
and `movemodelfn` for the model. For example `ToDevice(Flux.gpu, Flux.gpu)`,
moves them to a GPU if available. See [`ToGPU`](#).

By default, only moves `step.xs` and `step.ys`, but this can be extended
to other state by implementing `on(::StepBegin, ::MyCustomPhase, ::ToDevice, learner)`.
"""
struct ToDevice <: Callback
    movedatafn
    movemodelfn
end


function on(::EpochBegin, ::Phase, cb::ToDevice, learner)
    model!(learner, cb.movemodelfn(learner.model))
end

stateaccess(::ToDevice) = (
    model = Write(),
    params = Write(),
    step = Write(),
)

function on(::StepBegin, ::Phase, cb::ToDevice, learner)
    learner.step.xs = cb.movedatafn(learner.step.xs)
    learner.step.ys = cb.movedatafn(learner.step.ys)
end


function garbagecollect()
    GC.gc()
    if Base.Sys.islinux()
        ccall(:malloc_trim, Cvoid, (Cint,), 0)
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
    return throttle(
        CustomCallback((learner) -> garbagecollect(), StepEnd, Phase),
        StepEnd(),
        freq = nsteps)
end
