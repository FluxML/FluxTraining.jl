import Logging: with_logger
import ProgressMeter: Progress, next!
import TensorBoardLogger: TBLogger

# ProgressBarLogger

mutable struct ProgressBarLogger <: AbstractCallback
    p::Union{Nothing, Progress}
end
ProgressBarLogger() = ProgressBarLogger(nothing)

function on_epoch_begin(cb::ProgressBarLogger, state::TrainingState, phase::AbstractFittingPhase )
    cb.p = Progress(length(getdataloader(state.learner.databunch, phase)), "Epoch $(state.epoch) $(phase): ")
end

function on_batch_end(cb::ProgressBarLogger, state::TrainingState, phase::AbstractFittingPhase )
    next!(cb.p)
end


# DebugCallbackCallback

struct DebugCallbackCallback <: AbstractCallback end

on_train_begin(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_train_begin")
on_train_end(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_train_end")

on_epoch_begin(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_epoch_begin")
on_epoch_end(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_epoch_end")

on_batch_begin(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_batch_begin")
on_batch_end(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_batch_end")

on_loss_begin(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_loss_begin")

on_backward_begin(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_backward_begin")
on_backward_end(cb::DebugCallbackCallback, state::TrainingState, phase::AbstractFittingPhase ) = println("on_backward_end")


# TensorBoradLogger

# TODO: log more
struct TensorBoardCallback <: AbstractCallback
    logger::TBLogger
    TensorBoardCallback(logdir::AbstractString, runname::AbstractString) = new(
        TBLogger(joinpath(logdir, runname)))
end

function on_batch_end(cb::TensorBoardCallback, state::TrainingState, phase::AbstractFittingPhase )
    phasename = phase isa TrainingPhase ? "train" : "val"
    with_logger(cb.logger) do
        @info "$(phasename)/step" loss=state.lossbatch
    end
end


# PrintMetrics

struct PrintMetrics <: AbstractCallback end

function on_epoch_end(cb::PrintMetrics, state::TrainingState, phase::ValidationPhase)
    df = state.learner.recorder.epochdf
    dfsubset = df[(df.epoch .== state.epoch) .& (df.phase .== string(phase)), :]
    println(string([(row.metric, row.value) for row in eachrow(dfsubset)]...))
end


# StopOnNaNLoss

struct StopOnNaNLoss <: AbstractCallback end

struct NaNLossException <: Exception end

function on_backward_begin(cb::StopOnNaNLoss, state::TrainingState, phase::AbstractFittingPhase )
    !isnan(state.lossbatch) || throw(NaNLossException())
end
