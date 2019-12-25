import Logging: with_logger
import ProgressMeter: Progress, next!
import TensorBoardLogger: TBLogger

# ProgressBarLogger

mutable struct ProgressBarLogger <: AbstractCallback
    p::Union{Nothing, Progress}
end
ProgressBarLogger() = ProgressBarLogger(nothing)

function on_epoch_begin(cb::ProgressBarLogger, state::TrainingState)
    cb.p = Progress(length(state.data), "Epoch $(state.epoch): ")
end

function on_batch_end(cb::ProgressBarLogger, state::TrainingState)
    next!(cb.p)
end


# DebugCallbackCallback

struct DebugCallbackCallback <: AbstractCallback end

on_train_begin(c::DebugCallbackCallback, state::TrainingState) = println("on_train_begin")
on_train_end(c::DebugCallbackCallback, state::TrainingState) = println("on_train_end")

on_epoch_begin(c::DebugCallbackCallback, state::TrainingState) = println("on_epoch_begin")
on_epoch_end(c::DebugCallbackCallback, state::TrainingState) = println("on_epoch_end")

on_batch_begin(c::DebugCallbackCallback, state::TrainingState) = println("on_batch_begin")
on_batch_end(c::DebugCallbackCallback, state::TrainingState) = println("on_batch_end")

on_loss_begin(c::DebugCallbackCallback, state::TrainingState) = println("on_loss_begin")

on_backward_begin(c::DebugCallbackCallback, state::TrainingState) = println("on_backward_begin")
on_backward_end(c::DebugCallbackCallback, state::TrainingState) = println("on_backward_end")


# TensorBoradLogger

struct TensorBoardCallback <: AbstractCallback
    logger::TBLogger
    TensorBoardCallback(logdir::AbstractString, runname::AbstractString) = new(
        TBLogger(joinpath(logdir, runname)))
end

function on_batch_end(c::TensorBoardCallback, state::TrainingState)
    with_logger(c.logger) do
        @info "train/step" loss=state.lossbatch
    end
end
