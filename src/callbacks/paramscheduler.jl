
# Schedule data structures

@with_kw struct Schedules
    data::Dict = Dict()
end

@with_kw struct Schedule{T}
    nsteps::Int
    from::T
    to::T
    annealfn = anneal_linear
end


duration(schedule::Schedule) = schedule.nsteps
duration(schedules::Vector{Schedule}) = sum(duration.(schedules))
duration(schedules::Schedules) = maximum(duration.(values(schedules.data)))

function schedulevalue(schedule::Schedule, step)
    pctg = min(1, step / schedule.nsteps)
    return schedule.annealfn(pctg, schedule.from, schedule.to)
end

function schedulevalue(schedules::Vector{Schedule}, step)
    length(schedules) > 0 || error("Missing schedule")
    i = 1
    while duration(schedules[i]) < step
        step -= duration(schedules[i])
        i += 1
    end
    return schedulevalue(schedules[i], step)
end

# Scheduler Callback

struct ParamScheduler <: AbstractCallback end

order(::Type{ParamScheduler}) = -80

"""
    on(::BatchBegin, phase::AbstractTrainingPhase, cb::ParamScheduler, learner)

Every batch, set all hyperparameters according to `cb.scheduledict`.
"""
function on(::BatchBegin, phase::AbstractTrainingPhase, cb::ParamScheduler, learner)
    for (P, schedule) in learner.state.schedule.data
        setoptimparam!(
            learner.opt,
            P,
            # TODO: maybe add one? + 1
            schedulevalue(schedule, learner.state.history.nsteps + 1))
    end
end


# Sample schedules

function onecycleschedule(
        nsteps,
        maxlr;
        startfactor = 1/10,
        endfactor = 1/100,
        pctstart = 0.1)

    upschedule = Schedule(
        round(Int, nsteps * pctstart),
        maxlr * startfactor,
        maxlr,
        anneal_cosine)
    downschedule = Schedule(
        round(Int, nsteps * (1-pctstart)),
        maxlr,
        maxlr * endfactor,
        anneal_cosine)

    return Dict(LR => [upschedule, downschedule])
end



# Utilities
# TODO: update

#=
function sampleschedule(
        schedule::Union{ParamSchedule, AbstractVector{ParamSchedule}};
        epochs = duration(schedule))
    xs = collect(0:0.01:epochs)
    ys = map(xs) do x
        schedulevalue(schedule, x)
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
=#
