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
using DocStringExtensions
import OnlineStats
using OnlineStats: EqualWeight, Mean, OnlineStat
using Parameters
using ProgressMeter: Progress, next!
using Statistics: mean
using UUIDs
using Zygote
using Animations
using TensorBoardLogger
using Zygote: Grads, gradient
using ValueHistories
using DataStructures: DefaultDict


# functional
include("./functional/anneal.jl")
include("./functional/metrics.jl")

# utilities
include("./util/ioutils.jl")

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
include("./callbacks/callbacks.jl")
include("./callbacks/customcallback.jl")
include("./callbacks/metrics.jl")
include("./callbacks/recorder.jl")

# hyperparameter scheduling
include("./callbacks/hyperparameters.jl")
include("./callbacks/scheduler.jl")


# learner
include("./learner.jl")

include("./train.jl")


# TODO: remove old exports
export AbstractCallback,
    AbstractMetric,
    Accuracy,
    Loss,
    CancelBatchException,
    CancelEpochException,
    CancelFittingException,
    Checkpointer,
    CheckpointAny,
    CheckpointLowest,
    CustomCallback,
    DataLoader,
    EarlyStopping,
    ToGPU,
    GarbageCollect,
    Learner,
    Metric,
    Recorder,
    ProgressBarLogger,
    MetricsLogger,
    ParamSchedule,
    ParamScheduler,
    TrainingPhase,
    ValidationPhase,
    Schedule,
    Scheduler,
    StopOnNaNLoss,

    LearningRate,

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
