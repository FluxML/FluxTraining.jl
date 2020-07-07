"""
    fit!(learner, phases)
    fit!(learner, phase)
    fit!(learner, n)
"""
function fit!(learner::Learner, phases::AbstractVector{<:AbstractFittingPhase})::Learner
    handler = CallbackHandler(learner)
    try
        for phase in phases
            fitepoch!(learner, phase, handler)
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
function fitepoch!(learner, phase = learner.phase, handler = CallbackHandler(learner))
    learner.phase = phase
    try
        fitepochphase!(learner, phase, handler)
    catch e
        if e isa CancelEpochException
            @info "Epoch was cancelled" error=e
        else
            rethrow()
        end
    end

    return learner
end


function fitbatch!(learner, batch, phase = learner.phase, handler = CallbackHandler(learner))
    learner.phase = phase
    learner.batch = BatchState()
    try
        fitbatchphase!(learner, batch, phase, handler)
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
        cbhandler::CallbackHandler = CallbackHandler(learner))
    EpochBegin() |> cbhandler

    for batch in getdataloader(learner.databunch, phase)
        fitbatch!(learner, batch, phase, cbhandler)
    end

    EpochEnd() |> cbhandler
end


"""
    fitbatchphase!(learner, phase, batch, cbhandler)
"""
function fitbatchphase!(
        learner::Learner,
        batch,
        ::AbstractTrainingPhase,
        cbhandler::CallbackHandler)

    BatchBegin() |> cbhandler
    learner.batch.batch = x, y = learner.device(batch)


    gs = learner.batch.gradients = gradient(learner.params) do
        y_pred = learner.batch.y_pred = learner.model(x)

        LossBegin() |> cbhandler
        loss = learner.batch.loss = learner.lossfn(y_pred, y)

        BackwardBegin() |> cbhandler
        return loss
    end

    BackwardEnd() |> cbhandler

    # FIXME: revert to learner.batch.gradients
    update!(learner.opt, learner.params, gs)

    BatchEnd() |> cbhandler
    return learner
end


function fitbatchphase!(
        learner::Learner,
        batch,
        ::ValidationPhase,
        cbhandler::CallbackHandler = CallbackHandler(learner))
    BatchBegin() |> cbhandler
    learner.batch = BatchState()
    learner.batch.batch = x, y = learner.device(batch)

    y_pred = learner.batch.y_pred = learner.model(x)

    LossBegin() |> cbhandler
    loss = learner.batch.loss = learner.lossfn(y_pred, y)

    BatchEnd() |> cbhandler
end
