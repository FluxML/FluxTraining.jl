module FluxTraining

using BSON: @load, @save
using DataLoaders
using Flux
using Flux: onecold
using Flux.Optimise: update!
using Glob
using MLDataUtils: datasubset, nobs
import OnlineStats
using OnlineStats: EqualWeight, Mean, OnlineStat
using ProgressMeter: Progress, next!
using Statistics: mean
using UUIDs
using Zygote: Grads, gradient


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

# advanced
include("./advanced/lrfinder.jl")
include("./advanced/onecycleschedule.jl")

#include("./util/plotutils.jl")

# submodules
include("./models/Models.jl")

export
    Models,

    AbstractCallback,
    AbstractMetric,
    Accuracy,
    AverageLoss,
    Checkpointer,
    CheckpointAny,
    CheckpointLowest,
    CustomCallback,
    DataBunch,
    DataLoader,
    EarlyStopping,
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
    plotlosses,
    plotlrfinder,
    plotschedule,
    savemodel,
    saveweights,
    setschedule!,
    splitdataset,
    starttraining

end # module
