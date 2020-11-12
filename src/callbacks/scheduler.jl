"""
    Schedule(ts, values, interpolations; unit = :epoch; kwargs...)

Describes how the values of a hyperparameter change over the
training.

## Examples

```julia
# one-cycle schedule over 10 epochs from 0.01 over 0.1 to 0.001
Schedule([0, 2, 10], [0.01, 0.1, 0.001], [sineio(), sineio()])
```

`Schedule` is an alias for `Animations.Animation`, see the
[documentation](https://jkrumbiegel.github.io/Animations.jl/dev)
for more detailed information.
"""
const Schedule = Animation

"""
    Scheduler(schedules...)

Callback for hyperparameter scheduling. Takes pairs of [`HyperParameter`](#)
types and [`Schedule`](#)s.

## Example

```julia
lrschedule = Schedule([0, 10], [0.1, 0.001], Animations.sineio())
scheduler = Scheduler(
    LearningRate => lrschedule
)
```

See also [`Schedule`](#).
"""
struct Scheduler <: Callback
    schedules::Dict{Type{<:HyperParameter}, Animations.FiniteLengthAnimation}
    Scheduler(args...; unit = :epoch, kwargs...) = new(Dict(args...; kwargs...))
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
end


function on(::BatchBegin, ::AbstractTrainingPhase, scheduler::Scheduler, learner)
    step = learner.cbstate.history.steps + 1
    for (H, animation) in scheduler.schedules
        value = Animations.at(animation, step)
        sethyperparameter!(learner, H, value)
        push!(learner.cbstate.hyperparams, Symbol(H), step, value)
    end
end


"""
    onecycle(nepochs, max_val, [start_val, end_val; start_pctg])

Creates a one-cycle [`Schedule`](#) over `nepochs` epochs from `start_val`
over `max_val` to `end_val`.
"""
function onecycle(
        nepochs, max_val,
        start_val = max_val / 10,
        end_val = max_val / 30;
        start_pctg = 0.1)
    return Animation(
        [0, nepochs * start_pctg, nepochs],
        [start_val, max_val, end_val],
        [Animations.sineio(), Animations.sineio()]
    )

end
