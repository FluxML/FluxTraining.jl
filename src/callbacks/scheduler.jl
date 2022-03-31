"""
    Scheduler(schedules...)

Callback for hyperparameter scheduling. Takes pairs of [`HyperParameter`](#)
types and [`Schedule`](#)s.

See [the tutorial](/documents/docs/tutorials/hyperparameters.md) for more information.

## Example

```julia
es = length(learner.data.training)
lrschedule = ParameterSchedulers.Step(;λ=1.0, γ=0.9, step_sizes=[10, 20])
scheduler = Scheduler(
    LearningRate => lrschedule
)
```
"""
mutable struct Scheduler <: Callback
    schedules::Dict{Type{<:HyperParameter}, ParameterSchedulers.AbstractSchedule}
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


function init!(scheduler::Scheduler, learner)
    if !haskey(learner.cbstate, :hyperparams)
        learner.cbstate.hyperparams = MVHistory()
    end
    scheduler.step = 1
end


function on(::StepBegin, phase::AbstractTrainingPhase, scheduler::Scheduler, learner)
    step = scheduler.step
    for (H, schedule) in scheduler.schedules
        value = schedule(step)
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

## Examples

```julia

epochlength = length(traindataiter)
cb = Scheduler(LearningRate => onecycle(10epochlength, 0.01))
learner = Learner(<args>...; callbacks=[cb])
```
"""
function onecycle(
        nsteps, max_val;
        pct_start=0.25,
        div=25, divfinal=1e5,
        start_val=max_val / div, end_val=max_val / divfinal)

    Sequence([Sin(;λ0=max_val, λ1=start_val, period=ceil(Int, nsteps*pct_start)), CosAnneal(;λ0=end_val, λ1=max_val, period=floor(Int, (nsteps*(1-pct_start))), restart=false)], [nsteps*pct_start, nsteps * (1-pct_start)])
end
