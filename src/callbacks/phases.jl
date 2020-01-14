
abstract type AbstractFittingPhase end
abstract type AbstractTrainingPhase <: AbstractFittingPhase end

struct TrainingPhase <: AbstractTrainingPhase end


struct ValidationPhase <: AbstractFittingPhase end
struct TestPhase <: AbstractFittingPhase end
struct UninitializedPhase <: AbstractFittingPhase end

Base.show(io::IO, phase::TrainingPhase) = print(io, "Training")
Base.show(io::IO, phase::ValidationPhase) = print(io, "Validation")
Base.show(io::IO, phase::TestPhase) = print(io, "Test")
Base.show(io::IO, phase::UninitializedPhase) = print(io, "Uninitialized")


getdataloader(databunch::DataBunch, phase::AbstractTrainingPhase) = databunch.traindl
getdataloader(databunch::DataBunch, phase::ValidationPhase) = databunch.valdl
getdataloader(databunch::DataBunch, phase::TestPhase) = databunch.testdl
