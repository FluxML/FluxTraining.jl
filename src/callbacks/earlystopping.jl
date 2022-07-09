

# Early stopping
"""
    EarlyStopping(criteria...; kwargs...)
    EarlyStopping(n)

Stop training early when `criteria` are met. See
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl) for available stopping
criteria.

Passing an integer `n` uses the simple patience criterion: stop if the
validation loss hasn't decreased for `n` epochs.

You can control which phases are taken to measure the out-of-sample loss
and the training loss with keyword arguments `trainphase` (default
[`AbstractTrainingPhase`](#)) and `testphase` (default [`AbstractValidationPhase`](#)).

## Examples

```julia
Learner(model, lossfn, callbacks=[EarlyStopping(3)])
```

```julia
import FluxTraining.ES: Disjunction, InvalidValue, TimeLimit

callback = EarlyStopping(Disjunction(InvalidValue(), TimeLimit(0.5)))
Learner(model, lossfn, callbacks=[callback])
```
"""
mutable struct EarlyStopping <: Callback
    criterion::ES.StoppingCriterion
    state
    testphase::Type{<:Phase}
    trainphase::Type{<:Phase}
end

function EarlyStopping(
        criterion;
        testphase = AbstractValidationPhase,
        trainphase = AbstractTrainingPhase)
    if (testphase isa trainphase) || (trainphase isa testphase)
        error("`trainphase` and `testphase` must not be subtypes of one another")
    end
    return EarlyStopping(criterion, nothing, testphase, trainphase)
end

function EarlyStopping(n::Int; kwargs...)
    return EarlyStopping(ES.Patience(n); kwargs...)
end

Base.show(io::IO, cb::EarlyStopping) = print(io, "EarlyStopping(", cb.criterion, ")")

function on(::EpochEnd, phase::Phase, cb::EarlyStopping, learner)
    loss = last(learner.cbstate.metricsepoch[phase], :Loss)[2]

    if phase isa cb.testphase
        if isnothing(cb.state)
            cb.state = ES.EarlyStopping.update(cb.criterion, loss)
        else
            cb.state = ES.EarlyStopping.update(cb.criterion, loss, cb.state)
        end
    elseif phase isa cb.trainphase
        if isnothing(cb.state)
            cb.state = ES.EarlyStopping.update_training(cb.criterion, loss)
        else
            cb.state = ES.EarlyStopping.update_training(cb.criterion, loss, cb.state)
        end
    end
    if !isnothing(cb.state) && ES.EarlyStopping.done(cb.criterion, cb.state)
        throw(CancelFittingException(ES.EarlyStopping.message(cb.criterion, cb.state)))
    end
end


stateaccess(::EarlyStopping) = (cbstate = (metricsepoch = Read(),),)
runafter(::EarlyStopping) = (Metrics,)
