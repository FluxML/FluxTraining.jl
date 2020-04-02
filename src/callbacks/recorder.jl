"""
    Recorder(epoch, step, epochstats, stepstats)
"""
mutable struct Recorder <: AbstractCallback
    epoch::Int  # current or last completed epoch
    step::Int  # completed steps in current epoch
    steptotal::Int  # completed steps in total
    epochstats
    stepstats
end

Recorder() = Recorder(0, 0, 0, [], [])

order(cb::Type{Recorder}) = -90


function on(::EpochBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    recorder.epoch += 1
    recorder.step = 0
end

function on(::BatchBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    recorder.step += 1
    recorder.steptotal += 1
end


function on(::BatchEnd, phase::AbstractTrainingPhase, recorder::Recorder, learner)
    stepdata = Dict{String, Any}(
        "epochstep" => recorder.step,
        "step" => recorder.steptotal
    )

    for metric in learner.metrics
        stepdata["step/$metric"] = value(metric)
    end

    for (OptimParam, value) in getoptimparams(learner.opt)
        stepdata["opt/$(OptimParam.name)"] = value
    end

    push!(recorder.stepstats, stepdata)
end

function on(
        ::EpochEnd,
        phase::AbstractFittingPhase,
        recorder::Recorder,
        learner)
        
    epochdata = Dict(
        "phase" => string(typeof(phase).name),
        "epoch" => recorder.epoch
    )
    for metric in learner.metrics
        epochdata["$(string(typeof(phase).name))/$metric"] = Float64(value(metric))
    end

    push!(recorder.epochstats, epochdata)
end

