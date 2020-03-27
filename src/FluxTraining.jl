module FluxTraining

using DataLoaders
using Flux


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
include("./util/plotutils.jl")

# callback implementations
include("./callbacks/callbacks.jl")
include("./callbacks/experimenttracker.jl")
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

# submodules
include("./models/Models.jl")

export 
    Models,

    AbstractCallback,
    AbstractMetric,
    AverageLoss,
    Checkpointer,
    CheckpointAny,
    CheckpointLowest,
    CustomCallback,
    DataBunch,
    Learner,
    LRFinderPhase,
    ExperimentTracker,
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
