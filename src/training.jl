"""
    epoch!(learner, phase[, dataiter])

Train `learner` for one epoch on `dataiter`. Iterates through
`dataiter` and [`step!`](#)s for each batch/item.

If no data iterator is passed in, use `learner.data[phasedataiter(phase)]`.

## Extending

The default implementation iterates over every batch in `dataiter`
and calls [`step!`](#) for each. This behavior can be overloaded
by implementing `epoch!(learner, ::MyPhase, dataiter)`.

If you're implementing a custom `epoch!` method, it is recommended
you make use of [`runepoch`](#) to get begin and end events as well
as proper handling of [`CancelEpochException`](#)s.

See the default implementation for reference.
"""
function epoch!(learner, phase::Phase, dataiter=learner.data[phasedataiter(phase)])
    runepoch(learner, phase) do _
        for batch in dataiter
            step!(learner, phase, batch)
        end
    end
end


"""
    step!(learner, phase::Phase, batch)

Run one step of training for `learner` on batch.
Behavior is customized through `phase`.

## Extending

This is a required method for custom [`Phase`](#)s to implement.
To implement `step!`, it is recommended you make use of [`runstep`](#)
to get begin and end events as well as proper handling of
[`CancelStepException`](#)s.

See the implementations of [`TrainingPhase`](#) and [`ValidationPhase`](#)
for reference.
"""
function step! end


function step!(learner, phase::TrainingPhase, batch)
    xs, ys = batch
    runstep(learner, phase, (; xs=xs, ys=ys)) do handle, state

        state.grads = _gradient(learner.optimizer, learner.model, learner.params) do model
            state.ŷs = model(state.xs)
            handle(LossBegin())
            state.loss = learner.lossfn(state.ŷs, state.ys)
            handle(BackwardBegin())
            return state.loss
        end
        handle(BackwardEnd())
        learner.params, learner.model = _update!(
            learner.optimizer, learner.params, learner.model, state.grads)
    end
end


# Handle both old Flux.jl and new Optimisers.jl optimisers

_gradient(f, _, m, _) = gradient(f, m)[1]
_gradient(f, ::Flux.Optimise.AbstractOptimiser, m, ps::Params) = gradient(() -> f(m), ps)

function _update!(optimizer::Flux.Optimise.AbstractOptimiser, params, model, grads)
    update!(optimizer, params, grads)
    return params, model
end
function _update!(_, st, model, grads)
    st, model = Optimisers.update!(st, model, grads)
    return st, model
end


function step!(learner, phase::ValidationPhase, batch)
    xs, ys = batch
    runstep(learner, phase, (;xs=xs, ys=ys)) do _, state
        state.ŷs = learner.model(state.xs)
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end


"""
    runepoch(epochfn, learner, phase)

Run `epochfn` inside the context of an epoch. Calls `epochfn(handle)`
where `handle(e)` can be called to dispatch events.

Takes care of dispatching [`EpochBegin`](#) and [`EpochEnd`](#)
events as well as handling [`CancelEpochException`](#)s.

"""
function runepoch(epochfn, learner, phase::Phase)
    handlefn(e) = handle(learner.callbacks.runner, e, phase, learner)
    try
        handlefn(EpochBegin())
        epochfn(handlefn)
        handlefn(EpochEnd())
    catch e
        if e isa CancelEpochException
            @debug "Epoch skipped" error = e
            handlefn(EpochEnd())
        else
            rethrow()
        end
    end
end

"""
    runstep(stepfn, learner, phase) -> state

Run `stepfn` inside the context of a step. Calls `stepfn(handle, state)`
where `handle(e)` can be called to dispatch events and `state` is a [`PropDict`](#)
which step data, gradients and losses can be written to. Return `state`.

Takes care of dispatching [`StepBegin`](#) and [`StepEnd`](#)
events as well as handling [`CancelStepException`](#)s.
"""
function runstep(stepfn, learner, phase::Phase, initialstate = (;))
    state = PropDict(pairs(initialstate))
    handlefn(e) = handle(learner.callbacks.runner, e, phase, learner)
    try
        learner.step = state
        handlefn(StepBegin())
        stepfn(handlefn, state)
        handlefn(StepEnd())
        return state
    catch e
        if e isa CancelStepException
            @debug "Step skipped" error = e
        else
            rethrow()
        end
    end
    return state
end



# Utilities

"""
    fit!(learner, nepochs)
    fit!(learner, nepochs, (trainiter, validiter))

Train `learner` for `nepochs` of training and validation each. Use data
iterators that are passed in. If none are given, use `learner.data.training`
and `learner.data.validation`.

## Examples

```julia
fit!(learner, 10)
fit!(learner, 10, (traindl, valdl))
```

"""
function fit!(learner, nepochs::Int, (trainiter, validiter))
    for i in 1:nepochs
        epoch!(learner, TrainingPhase(), trainiter)
        epoch!(learner, ValidationPhase(), validiter)
    end
end

function fit!(learner, nepochs::Int)
    fit!(learner, nepochs, (learner.data.training, learner.data.validation))
end
