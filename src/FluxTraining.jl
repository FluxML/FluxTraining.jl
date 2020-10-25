module FluxTraining

#=
Refactoring:
- better serialization of Learners
=#


using LightGraphs
using BSON: @load, @save
using Flux
using Flux: Params, onecold
using Flux.Optimise: update!
using Glob
import OnlineStats
using OnlineStats: EqualWeight, Mean, OnlineStat
using Parameters
using ProgressMeter: Progress, next!
using Statistics: mean
using UUIDs
using Zygote
using Animations
using TensorBoardLogger: TBLogger, log_value, log_image, log_text, log_histogram
using Zygote: Grads, gradient
using ValueHistories
using DataStructures: DefaultDict


# functional
include("./functional/anneal.jl")
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


# callback implementations
include("./callbacks/conditional.jl")
include("./callbacks/callbacks.jl")
include("./callbacks/custom.jl")
include("./callbacks/metrics.jl")
include("./callbacks/recorder.jl")
include("./callbacks/checkpointer.jl")

# hyperparameter scheduling
include("./callbacks/hyperparameters.jl")
include("./callbacks/scheduler.jl")


# learner
include("./learner.jl")

include("./train.jl")


# TODO: remove old exports
export AbstractCallback,
    Loss,
    ConditionalCallback,
    CancelBatchException,
    CancelEpochException,
    CancelFittingException,
    Checkpointer,
    CustomCallback,
    EarlyStopping,
    ToGPU,
    GarbageCollect,
    Learner,
    Metric,
    Recorder,
    ProgressPrinter,
    Metrics,
    MetricsPrinter,
    ParamSchedule,
    ParamScheduler,
    TrainingPhase,
    ValidationPhase,
    Schedule,
    Scheduler,
    Logger,
    TensorBoardBackend,
    StopOnNaNLoss,

    LearningRate,

    throttle,
    accuracy,
    anneal_linear,
    anneal_cosine,
    anneal_exp,
    anneal_const,
    endtraining,
    fit!,
    loadmodel,
    loadweights,
    onecycle,
    savemodel,
    saveweights,
    setschedule!,
    splitdataset,
    starttraining

end # module
