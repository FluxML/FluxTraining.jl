using Plots: plot, plot!


struct ParamSchedule
    nepochs::Real
    startvalue::Real
    endvalue::Real
    anneal_fn
end
ParamSchedule(n, s, e; anneal_fn = anneal_linear) = ParamSchedule(n, s, e, anneal_fn)

duration(sched::ParamSchedule) = sched.nepochs
duration(scheds::AbstractVector{ParamSchedule}) = sum(duration.(scheds))

struct ParamScheduler <: AbstractCallback
    scheduledict::Dict{Type{<:OptimParam}, AbstractVector{ParamSchedule}}
end

"""
    on(::BatchBegin, phase::AbstractTrainingPhase, cb::ParamScheduler, learner)

Every batch, set all hyperparameters according to `cb.scheduledict`.
"""
function on(::BatchBegin, phase::AbstractTrainingPhase, cb::ParamScheduler, learner)
    e, s = learner.recorder.epoch, learner.recorder.step
    epochs = (e - 1) + (s / numsteps(learner, phase))

    for (P, schedules) in cb.scheduledict
        val = schedulevalue(schedules, epochs)
        setoptimparam!(learner.opt, P, schedulevalue(schedules, epochs))
    end
end


function schedulevalue(schedule::ParamSchedule, epochs::Real)::Union{Nothing, Real}
    pctg = min(epochs / schedule.nepochs, 1)  # return last value if out of bounds
    return schedule.anneal_fn(pctg, schedule.startvalue, schedule.endvalue)
end


function schedulevalue(schedules::AbstractVector{ParamSchedule}, epochs::Real)
    length(schedules) > 0 || error("Missing schedule")

    s = schedules[1]

    if epochs < s.nepochs || length(schedules) == 1
        return schedulevalue(s, epochs)
    else
        return schedulevalue(schedules[2:end], epochs - s.nepochs)
    end
end


# Utilities

function sampleschedule(
        schedule::Union{ParamSchedule, AbstractVector{ParamSchedule}};
        epochs = Training.duration(schedule))
    xs = collect(0:0.01:epochs)
    ys = map(xs) do x
        Training.schedulevalue(schedule, x)
    end
    xs, ys
end


function delayschedule(schedule::Dict, nepochs::Real)
    newschedule = Dict()
    for (P, schedules) in schedule
        v = schedules[1].startvalue
        delay = ParamSchedule(nepochs, v, v, anneal_const)
        newschedule[P] = vcat(delay, schedules)
    end
    return newschedule
end
