module Training

include("./dataloader.jl")
include("./databunch.jl")
include("./optimizers.jl")

# Callbacks
include("./functional/anneal.jl")
include("./functional/metrics.jl")
include("./callbacks/phases.jl")
include("./callbacks/callback.jl")
include("./callbacks/callbacks.jl")
include("./callbacks/metrics.jl")
include("./callbacks/recorder.jl")
include("./callbacks/paramscheduler.jl")

include("./learner.jl")
include("./train.jl")
include("./transform/functional.jl")
include("./transform/transform.jl")
include("./modelutils/initialization.jl")
include("./data/utils.jl")

# advanced
include("./advanced/lrfinder.jl")
include("./advanced/onecycleschedule.jl")


# submodules
include("./models/Models.jl")

export DataLoader,
    Models,

    DataBunch,
    Learner,
    DataLoader,
    AbstractCallback,
    AbstractMetric,
    ProgressBarLogger,
    PrintMetrics,
    ParamScheduler,
    AverageLoss,
    AverageMetric,

    accuracy,
    anneal_linear,
    anneal_cos,
    anneal_exp,
    anneal_const,
    computestats,
    getbatch,
    fit!,
    fitepoch!,
    fitbatch!,
    LRFinderPhase,
    onecycleschedule


end # module
