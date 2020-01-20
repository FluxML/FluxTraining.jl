import DataStructures: DefaultDict

"""
    Recorder(epoch, step, epochstats, stepstats)
"""
mutable struct Recorder <: AbstractCallback
    epoch::Integer  # completed training epochs
    step::Integer  # completed steps in current epoch
    epochstats
    stepstats
end

Recorder() = Recorder(0, 0, [], Dict())

order(cb::Type{Recorder}) = -50


function on(::EpochBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    recorder.epoch += 1
    recorder.step = 0
end

function on(::BatchBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    recorder.step += 1
end


function on(::BatchEnd, phase::AbstractTrainingPhase, recorder::Recorder, learner)
    for metric in learner.metrics
        k = string(metric)
        if !haskey(recorder.stepstats, k)
            recorder.stepstats[k] = []
        end
        xs = recorder.stepstats[k]
        push!(xs, value(metric))
    end

    if !haskey(recorder.stepstats, LR)
        recorder.stepstats[LR] = []
    end
    xs = recorder.stepstats[LR]
    push!(xs, getoptimparam(learner.opt, LR))
end

function on(
        ::EpochEnd,
        phase::AbstractFittingPhase,
        recorder::Recorder,
        learner)

    for metric in learner.metrics
        row = [learner.recorder.epoch, string(phase), string(metric), value(metric)]
        push!(recorder.epochstats, row)
    end
end

getoptstats(opt) = Dict(LR => getoptimparam(opt, LR))
