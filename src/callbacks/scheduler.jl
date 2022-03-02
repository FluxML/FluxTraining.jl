"""
    Schedule(ts, values, interpolations; unit = :epoch; kwargs...)

Describes how the values of a hyperparameter change over the
training.

## Examples

```julia
# one-cycle schedule over 10 epochs from 0.01 over 0.1 to 0.001
es = length(learner.data.training)
Schedule([0es, 2es, 10es], [0.01, 0.1, 0.001], [sineio(), sineio()])
```

`Schedule` is an alias for `Animations.Animation`, see the
[Animations.jl documentation](https://jkrumbiegel.github.io/Animations.jl/dev)
for more detailed information.
"""
const Schedule = Animation

"""
    Scheduler(schedules...)

Callback for hyperparameter scheduling. Takes pairs of [`HyperParameter`](#)
types and [`Schedule`](#)s.

See [the tutorial](/documents/docs/tutorials/hyperparameters.md) for more information.

## Example

```julia
es = length(learner.data.training)
lrschedule = Schedule([0es, 10es], [0.1, 0.001], Animations.sineio())
scheduler = Scheduler(
    LearningRate => lrschedule
)
```

See also [`Schedule`](#).
"""
mutable struct Scheduler <: Callback
    schedules::Dict{Type{<:HyperParameter},Animations.FiniteLengthAnimation}
    step::Int
    Scheduler(args...; kwargs...) = new(Dict(args...; kwargs...), 1)
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
        data = Read(), cbstate = (; hyperparams=Write(), history = Read()),
        hpstateaccess...
    )
end


function init!(::Scheduler, learner)
    if !haskey(learner.cbstate, :hyperparams)
        learner.cbstate.hyperparams = MVHistory()
    end
end


function on(::StepBegin, phase::AbstractTrainingPhase, scheduler::Scheduler, learner)
    step = scheduler.step
    for (H, animation) in scheduler.schedules
        value = Animations.at(animation, scheduler.step)
        sethyperparameter!(learner, H, value)
        push!(
            learner.cbstate.hyperparams,
            Symbol(H),
            learner.cbstate.history[phase].steps,
            value)
    end
    scheduler.step += 1
end


"""
    onecycle(nsteps, max_val, [start_val, end_val; pct_start])

Creates a one-cycle [`Schedule`](#) over `nsteps` steps from `start_val`
over `max_val` to `end_val`.
"""
function onecycle(
        nsteps, max_val;
        pct_start=0.25,
        div=25, divfinal=1e5,
        start_val=max_val / div, end_val=max_val / divfinal)
    return Animation(
        [0, nsteps * pct_start, nsteps],
        [start_val, max_val, end_val],
        [Animations.sineio(), Animations.sineio()]
    )
end
