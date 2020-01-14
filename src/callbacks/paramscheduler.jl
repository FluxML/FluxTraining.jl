struct ParamSchedule
    nepochs::Real
    startvalue::Real
    endvalue::Real
    anneal_fn
end
ParamSchedule(n, s, e; anneal_fn = anneal_linear) = ParamSchedule(n, s, e, anneal_fn)


struct ParamScheduler <: AbstractCallback
    scheduledict::Dict{Type{<:OptimParam}, AbstractVector{ParamSchedule}}
end


function on_batch_begin(cb::ParamScheduler, state::TrainingState, phase::AbstractTrainingPhase)
    # completed epochs
    epochs = (state.epoch - 1) + (state.step / length(getdataloader(state.learner.databunch, phase)))

    for (P, schedules) in cb.scheduledict
        val = schedulevalue(schedules, epochs)
        if val !== nothing
            setoptimparam!(state.learner.opt, P, schedulevalue(schedules, epochs))
        end
    end
end

function schedulevalue(schedule::ParamSchedule, epochs::Real)::Union{Nothing, Real}
    pctg = epochs / schedule.nepochs
    if pctg > 1
        return nothing
    else
        return schedule.anneal_fn(pctg, schedule.startvalue, schedule.endvalue)
    end
end


function schedulevalue(schedules::AbstractVector{ParamSchedule}, epochs::Real)
    length(schedules) > 0 || return nothing

    s = schedules[1]

    if epochs < s.nepochs
        return schedulevalue(s, epochs)
    else
        return schedulevalue(schedules[2:end], epochs - s.nepochs)
    end
end
