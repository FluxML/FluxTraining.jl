"""
    Schedule(ts, values, interpolations; unit = :epoch; kwargs...)

Describes how the values of a hyperparameter change over the
training.

## Examples

```julia
# one-cycle schedule over 10 epochs from 0.01 over 0.1 to 0.001
Schedule([0, 2, 10], [0.01, 0.1, 0.001], [sineio(), sineio()])
```

`Schedule` is a wrapper around `Animations.Animation`, see the
[documentation](https://jkrumbiegel.github.io/Animations.jl/dev)
for more detailed information.
"""
struct Schedule{T}
    animation::Animation{T}
    unit::Symbol
end


Schedule(args...; unit = :epoch, kwargs...) =
    Schedule(Animation(args...; kwargs...), unit)

struct Scheduler <: Callback
    schedules::Dict{Type{<:HyperParameter}, Schedule}
end

Base.show(io::IO, scheduler::Scheduler) = print(
    io, "Scheduler(", join(keys(scheduler.schedules), ", "), ")")



function stateaccess(scheduler::Scheduler)
    # TODO: implement proper merging of permissions
    if length(keys(scheduler.schedules)) == 0
        hpstateaccess = (;)
    else
        hpstateaccess = merge(stateaccess.(keys(scheduler.schedules))...)
    end
    return (
        data = Read(), cbstate = (history = Read(), hyperparams = Write()),
        hpstateaccess...
    )
end


function on(::Init, phase, scheduler::Scheduler, learner)
    learner.cbstate.hyperparams = MVHistory()
    # convert schedules to use step as a unit
    epochlength = length(learner.data[1])
    for (H, schedule) in scheduler.schedules
        if schedule.unit == :epoch
            scheduler.schedules[H] = Schedule(schedule.animation * epochlength, :step)
        end
    end
end


function on(::BatchBegin, ::AbstractTrainingPhase, scheduler::Scheduler, learner)
    step = learner.cbstate.history.steps
    for (H, schedule) in scheduler.schedules
        value = Animations.at(schedule.animation, step)
        sethyperparameter!(learner, H, Animations.at(schedule.animation, step))
        push!(learner.cbstate.hyperparams, Symbol(H), step, value)
    end
end


function onecycle(
        epochs, max_lr,
        start_lr = max_lr / 10,
        end_lr = max_lr / 30,
        start_pctg = 0.1)
    return Schedule(
        [0, epochs * start_pctg, epochs],
        [start_lr, max_lr, end_lr],
        [Animations.sineio(), Animations.sineio()]
    )

end
