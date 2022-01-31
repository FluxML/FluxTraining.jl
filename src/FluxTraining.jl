module FluxTraining


using LightGraphs
using BSON: @load, @save
using Flux
using Flux: Params, onecold
using Flux.Optimise: update!
using ImageCore
using Glob
module ES
    using Reexport
    @reexport using EarlyStopping
end
import OnlineStats
using FluxMPI
using MPI
using OnlineStats: EqualWeight, Mean, OnlineStat
using Parameters
using ProgressMeter: Progress, next!
using Statistics: mean
using UUIDs
using Zygote
using Animations
using TensorBoardLogger: TBLogger, log_value, log_image, log_text, log_histogram, tb_overwrite
using Zygote: Grads, gradient
using ValueHistories
using DataStructures: DefaultDict
using PrettyTables
using Dates


# functional
include("./functional/metrics.jl")

# callback system
include("./callbacks/protect.jl")
include("./callbacks/phases.jl")
include("./callbacks/events.jl")
include("./callbacks/callback.jl")
include("./callbacks/graph.jl")
include("./callbacks/execution.jl")

# logging
include("./callbacks/logging/Loggables.jl")
include("./callbacks/logging/logger.jl")
include("./callbacks/logging/tensorboard.jl")
include("./callbacks/logging/checkpointer.jl")


# callback implementations
include("./callbacks/conditional.jl")
include("./callbacks/callbacks.jl")
include("./callbacks/earlystopping.jl")
include("./callbacks/custom.jl")
include("./callbacks/metrics.jl")
include("./callbacks/recorder.jl")
include("./callbacks/sanitycheck.jl")
include("./callbacks/timer.jl")

# hyperparameter scheduling
include("./callbacks/hyperparameters.jl")
include("./callbacks/scheduler.jl")


# learner
include("./learner.jl")
include("./callbackutils.jl")

include("./training.jl")


export AbstractCallback,
    Loss,
    ConditionalCallback,
    CancelStepException,
    CancelEpochException,
    CancelFittingException,
    Checkpointer,
    TopKCheckpointer,
    CustomCallback,
    EarlyStopping,
    ToDevice,
    ToGPU,
    GarbageCollect,
    Learner,
    Metric,
    Recorder,
    ProgressPrinter,
    Metrics,
    MetricsPrinter,
    CSVMetricsLogger,
    TrainingPhase,
    ValidationPhase,
    Schedule,
    Scheduler,
    LogMetrics,
    SmoothLoss,
    LogHistograms,
    LogHyperParams,
    LogVisualization,
    TensorBoardBackend,
    StopOnNaNLoss,
    LearningRate,
    throttle,
    SanityCheck,
    TimingRecorder,
    accuracy,
    fit!,
    epoch!,
    step!,
    onecycle,
    loadmodel,
    savemodel
end # module
