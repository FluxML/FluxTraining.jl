import Logging: with_logger
import ProgressMeter: Progress, next!
import TensorBoardLogger: TBLogger

# ProgressBarLogger

mutable struct ProgressBarLogger <: AbstractCallback
    p::Union{Nothing,Progress}
end
ProgressBarLogger() = ProgressBarLogger(nothing)

function on(::EpochBegin,
        phase::AbstractFittingPhase,
        cb::ProgressBarLogger,
        learner)
    e = learner.recorder.epoch
    cb.p = Progress(numsteps(learner, phase), "Epoch $(e) $(phase): ")
end

on(::BatchEnd, ::AbstractFittingPhase, cb::ProgressBarLogger, learner) = next!(cb.p)


# TensorBoradLogger

# TODO: factor out into different package
struct TensorBoardCallback <: AbstractCallback
    logger::TBLogger
    TensorBoardCallback(logdir::AbstractString, runname::AbstractString) = new(TBLogger(joinpath(logdir, runname)))
end

function on(::BatchEnd, phase::AbstractFittingPhase, cb::TensorBoardCallback, learner)
    with_logger(cb.logger) do
        @info "$(typeof(phase).name)/step" loss = learner.batch.loss
    end
end


# PrintMetrics

struct PrintMetrics <: AbstractCallback end

function on(::EpochEnd,
        phase::AbstractFittingPhase,
        cb::PrintMetrics,
        learner)
    for metric in learner.metrics
        println(string(metric), ": ", value(metric))
    end
end


# StopOnNaNLoss

struct StopOnNaNLoss <: AbstractCallback end

struct NaNLossException <: Exception end

function on(::BackwardBegin, ::AbstractTrainingPhase, ::StopOnNaNLoss, learner)
    !isnan(learner.batch.loss) || throw(NaNLossException())
end
