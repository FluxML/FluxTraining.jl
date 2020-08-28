"""
    fit!(learner, phases)
    fit!(learner, phase)
    fit!(learner, n)
"""
function fit!(learner::Learner, phases::AbstractVector{<:AbstractFittingPhase})::Learner
    try
        for phase in phases
            fitepoch!(learner, phase)
        end
    catch e
        if e isa CancelFittingException
            @info "Fitting was cancelled" error=e
        else
            rethrow()
        end
    end
    return learner
end

fit!(learner::Learner, phase::AbstractFittingPhase)::Learner = fit!(learner, [phase])
fit!(learner, n::Int)::Learner = fit!(learner, repeat([TrainingPhase(), ValidationPhase()], n))


"""
    fitepoch!(learner, phase = learner.phase)
"""
function fitepoch!(learner, phase = learner.phase)
    learner.state.phase = phase
    try
        fitepochphase!(learner, phase)
    catch e
        if e isa CancelEpochException
            @info "Epoch was cancelled" error=e
        else
            rethrow()
        end
    end

    return learner
end


function fitbatch!(learner, batch, phase = learner.phase)
    learner.state.phase = phase
    learner.state.batch = BatchState()
    try
        fitbatchphase!(learner, batch, phase)
    catch e
        if e isa CancelBatchException
            @info "Batch was cancelled" error=e
        else
            rethrow()
        end
    end
end

"""
    fitepoch!(learner, phase)

Fit `learner` for one epoch.

Customize by deriving custom phase from `AbstractFittingPhase`.
"""
function fitepochphase!(
        learner::Learner,
        phase::Union{TrainingPhase, ValidationPhase, TestPhase},
    )
    handle(EpochBegin(), learner)

    for batch in getdataloader(phase, learner)
        fitbatch!(learner, batch, phase)
    end

    handle(EpochEnd(), learner)
end


function fitbatchphase!(
        learner::Learner,
        batch,
        ::AbstractTrainingPhase,
    )

    b = learner.state.batch
    b.xs, b.ys = batch

    handle(BatchBegin(), learner)

    b.grads = gradient(learner.state.params) do
        b.ŷs = learner.model(b.xs)

        handle(LossBegin(), learner)
        b.loss = learner.lossfn(b.ŷs, b.ys)

        handle(BackwardBegin(), learner)
        return b.loss
    end
    handle(BackwardEnd(), learner)

    update!(learner.opt, learner.state.params, b.grads)

    handle(BatchEnd(), learner)
    return learner
end


function fitbatchphase!(
        learner::Learner,
        batch,
        ::ValidationPhase,
    )
    b = learner.state.batch
    b.xs, b.ys = batch
    handle(BatchBegin(), learner)

    b.ŷs = learner.model(b.xs)

    handle(LossBegin(), learner)
    b.loss = learner.lossfn(b.ŷs, b.ys)

    handle(BatchEnd(), learner)
end
