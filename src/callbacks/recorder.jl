import DataStructures: DefaultDict
import DataFrames: DataFrame

mutable struct StepStats
    step
    optstats
    losses
end

StepStats() = StepStats(0, Vector{Dict}(), Vector{Float32}())

mutable struct Recorder <: AbstractCallback
    stepstats::StepStats
    epochdf
    Recorder() = new(
        StepStats(),
        DataFrame(
            [Integer, AbstractString, AbstractString, AbstractFloat],
            [:epoch, :phase, :metric, :value]
        ),
    )
end


order(cb::Type{Recorder}) = -50


function on_batch_end(recorder::Recorder, state::TrainingState, phase::AbstractTrainingPhase)
    push!(recorder.stepstats.optstats, getoptstats(state.learner.opt))
    push!(recorder.stepstats.losses, state.lossbatch)
    recorder.stepstats.step += 1
end

function on_epoch_end(recorder::Recorder, state::TrainingState, phase::AbstractFittingPhase)
    for metric in state.learner.metrics
        row = [state.epoch, string(phase), string(metric), value(metric)]
        push!(recorder.epochdf, row)
    end
end


getoptstats(opt) = Dict(LR => getoptimparam(opt, LR))
