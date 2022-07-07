module FluxTraining


using Graphs
using BSON: @load, @save
using Flux
using Flux: Params, onecold
using Flux.Optimise: update!
using ImageCore
using InlineTest
using Glob
module ES
    using Reexport
    @reexport using EarlyStopping
end
import OnlineStats
using OnlineStats: EqualWeight, Mean, OnlineStat
import Optimisers
using Parameters
using ProgressMeter: Progress, next!
using Statistics: mean
using UUIDs
using Zygote
using ChainRulesCore
using ParameterSchedulers
using TensorBoardLogger: TBLogger, log_value, log_image, log_text, log_histogram, tb_overwrite
using Zygote: Grads, gradient
using ValueHistories
using DataStructures: DefaultDict
using PrettyTables
using Setfield: @set

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

# hyperparameter scheduling
include("./callbacks/hyperparameters.jl")
include("./callbacks/scheduler.jl")


# learner
include("./learner.jl")
include("./callbackutils.jl")

include("./training.jl")

include("testutils.jl")

export AbstractCallback,
    Loss,
    ConditionalCallback,
    CancelStepException,
    CancelEpochException,
    CancelFittingException,
    Checkpointer,
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
    accuracy,
    fit!,
    epoch!,
    step!,
    onecycle,
    loadmodel,
    savemodel
end # module
