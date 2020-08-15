module FluxTraining

#=
Refactoring:
- better serialization of Learners
=#


using BSON: @load, @save
using DataLoaders
using Flux
using Flux: Params, onecold
using Flux.Optimise: update!
using Glob
using DocStringExtensions
using MLDataUtils: datasubset, nobs
import OnlineStats
using OnlineStats: EqualWeight, Mean, OnlineStat
using Parameters
using ProgressMeter: Progress, next!
using Statistics: mean
using UUIDs
using Zygote: Grads, gradient
using ValueHistories
using DataStructures: DefaultDict


# optimizers
include("./optimizers.jl")

# functional
include("./functional/anneal.jl")
include("./functional/metrics.jl")

# callback framework
include("./callbacks/callback.jl")

# utilities
include("./util/datautils.jl")
include("./util/ioutils.jl")

# callback implementations
include("./callbacks/callbacks.jl")
include("./callbacks/customcallback.jl")
include("./callbacks/checkpointer.jl")
include("./callbacks/metrics.jl")
include("./callbacks/recorder.jl")
include("./callbacks/paramscheduler.jl")

# learner
include("./learner.jl")

include("./train.jl")
include("./util/trainutils.jl")

# advanced
#include("./advanced/lrfinder.jl")
#include("./advanced/onecycleschedule.jl")


# submodules
include("./models/Models.jl")

export Models,

    AbstractCallback,
    AbstractMetric,
    Accuracy,
    AverageLoss,
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
    GarbageCollect,
    Learner,
    LRFinderPhase,
    Metric,
    MeanMetric,
    ProgressBarLogger,
    PrintMetrics,
    ParamSchedule,
    ParamScheduler,
    TrainingPhase,
    ValidationPhase,
    Schedule,
    Schedules,

    accuracy,
    anneal_linear,
    anneal_cosine,
    anneal_exp,
    anneal_const,
    endtraining,
    fit!,
    loadmodel,
    loadweights,
    onecycleschedule,
    savemodel,
    saveweights,
    setschedule!,
    splitdataset,
    starttraining

end # module
