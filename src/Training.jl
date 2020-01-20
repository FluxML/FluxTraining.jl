module Training

include("./utils.jl")

# data
include("./dataloader.jl")
include("./databunch.jl")
include("./data/utils.jl")
include("./optimizers.jl")

# functional
include("./functional/anneal.jl")
include("./functional/metrics.jl")


# callback framework
include("./callbacks/phases.jl")
include("./callbacks/callback.jl")
include("./callbacks/events.jl")
include("./callbacks/callbackhandler.jl")


# callback implementations
include("./callbacks/callbacks.jl")
include("./callbacks/metrics.jl")
include("./callbacks/recorder.jl")
include("./callbacks/paramscheduler.jl")

# learner
include("./learner.jl")

include("./train.jl")

# transform
include("./transform/functional.jl")
include("./transform/transform.jl")



# deprecated
include("./modelutils/initialization.jl")


# advanced
include("./advanced/lrfinder.jl")
include("./advanced/onecycleschedule.jl")

# introspection
include("./plot.jl")

# submodules
include("./models/Models.jl")

export DataLoader,
    Models,

    AbstractCallback,
    AbstractMetric,
    AverageLoss,
    Compose,
    DataBunch,
    DataLoader,
    Learner,
    LRFinderPhase,
    Normalize,
    OneHot,
    ProgressBarLogger,
    PrintMetrics,
    ParamSchedule,
    ParamScheduler,
    RandomResizedCrop,
    SampleTransform,
    TrainingPhase,
    ValidationPhase,

    accuracy,
    anneal_linear,
    anneal_cosine,
    anneal_exp,
    anneal_const,
    computestats,
    fit!,
    fitepoch!,
    fitbatch!,
    getbatch,
    imagetotensor,
    normalize,
    onecycleschedule,
    plotlosses,
    plotlrfinder,
    plotschedule,
    setschedule!,
    splitdataset,
    tensortoimage

end # module
