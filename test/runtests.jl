include("./imports.jl")


@testset ExtendedTestSet "FluxTraining.jl" begin
    include("./metrics.jl")
    include("./protected.jl")
    include("./training.jl")
    include("./callbacks/stoponnanloss.jl")
    include("./callbacks/custom.jl")
    include("./callbacks/conditional.jl")
    include("./callbacks/logging.jl")
    include("./callbacks/recorder.jl")
    include("./callbacks/scheduler.jl")
    include("./callbacks/checkpointer.jl")
    include("./callbacks/garbagecollect.jl")
    include("./callbacks/sanitycheck.jl")
    include("./callbacks/earlystopping.jl")
    include("./profilerunner.jl")
    include("./callbackutils.jl")
end
