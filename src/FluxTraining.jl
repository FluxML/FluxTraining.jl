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
using Zygote: Grads, gradient
using ValueHistories
using DataStructures: DefaultDict


# functional
include("./functional/anneal.jl")
include("./functional/metrics.jl")

# callback framework
include("./callbacks/protect.jl")
include("./callbacks/phases.jl")
include("./callbacks/events.jl")
include("./callbacks/callback.jl")
include("./callbacks/graph.jl")
include("./callbacks/execution.jl")
include("./callbacks/hyperparameters.jl")

# utilities
include("./util/ioutils.jl")

# callback implementations
include("./callbacks/callbacks.jl")
include("./callbacks/customcallback.jl")
include("./callbacks/metrics.jl")
include("./callbacks/recorder.jl")
include("./callbacks/scheduler.jl")

include("./callbacks/logging/checkpointer.jl")
include("./callbacks/logging/artifacts.jl")

# learner
include("./learner.jl")

include("./train.jl")
include("./util/trainutils.jl")


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
    DataBunch,
    DataLoader,
    EarlyStopping,
    ToGPU,
    GarbageCollect,
    Learner,
    LRFinderPhase,
    Metric,
    MeanMetric,
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
