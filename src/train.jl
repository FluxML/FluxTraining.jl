function initlearner!(learner, phases)
    if !(learner.callbacks.initialized)
        handle(Init(), learner, phases[1])
        learner.callbacks.initialized = true
    end
end
initlearner!(learner, phase::Phase) = initlearner!(learner, (phase,))
"""
    fit!(learner, phases)

Fit `learner` with [`Phase`](#)s `phases`. See [`./docs/training/basics.md`] for more
info on the training loop.
"""
function fit!(learner::Learner, phases::AbstractVector{<:Phase})
    initlearner!(learner, phases)
    try
        for phase in phases
            fitepoch!(learner, phase)
        end
    catch e
        if e isa CancelFittingException
            @debug "Fitting was cancelled" error=e
        else
            rethrow()
        end
    end
    return learner
end

fit!(learner::Learner, phase::Phase)::Learner = fit!(learner, [phase])

"""
    fit!(learner, n)

Shorthand for `fit!(learner, repeat([TrainingPhase(), ValidationPhase()], n))`, i.e.
a very basic training loop of `n` epochs of training followed by validation.
"""
fit!(learner, n::Int)::Learner = fit!(learner, repeat([TrainingPhase(), ValidationPhase()], n))


function fitepoch!(learner, phase)
    try
        fitepochphase!(learner, phase)
    catch e
        if e isa CancelEpochException
            @debug "Epoch was cancelled" error=e
        else
            rethrow()
        end
    end

    return learner
end


function fitbatch!(learner, batch, phase)
    learner.step = StepState()
    try
        fitbatchphase!(learner, batch, phase)
    catch e
        if e isa CancelStepException
            @debug "Step was cancelled" error=e
        else
            rethrow()
        end
    end
end

"""
    fitepoch!(learner, phase)

Fit `learner` for one epoch.

Customize by deriving custom phase from `Phase`.
"""
function fitepochphase!(
        learner::Learner,
        phase::Union{TrainingPhase, ValidationPhase, TestPhase},
    )
    dataiter = getdataiter(phase, learner)
    if dataiter === nothing
        throw(CancelEpochException("No data found for phase $(typeof(phase))"))
    end

    handle(EpochBegin(), learner, phase)

    for batch in dataiter
        fitbatch!(learner, batch, phase)
    end

    handle(EpochEnd(), learner, phase)
end


function fitbatchphase!(
        learner::Learner,
        batch,
        phase::AbstractTrainingPhase,
    )

    b = learner.step
    b.xs, b.ys = batch

    handle(StepBegin(), learner, phase)

    b.grads = gradient(learner.params) do
        b.ŷs = learner.model(b.xs)

        handle(LossBegin(), learner, phase)
        b.loss = learner.lossfn(b.ŷs, b.ys)

        handle(BackwardBegin(), learner, phase)
        return b.loss
    end
    handle(BackwardEnd(), learner, phase)

    update!(learner.optimizer, learner.params, b.grads)

    handle(StepEnd(), learner, phase)
    return learner
end


function fitbatchphase!(
        learner::Learner,
        batch,
        phase::ValidationPhase,
    )
    b = learner.step
    b.xs, b.ys = batch
    handle(StepBegin(), learner, phase)

    b.ŷs = learner.model(b.xs)

    handle(LossBegin(), learner, phase)
    b.loss = learner.lossfn(b.ŷs, b.ys)

    handle(StepEnd(), learner, phase)
end
