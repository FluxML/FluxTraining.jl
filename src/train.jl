using Flux: params, gpu
using Flux.Optimise: update!
using Zygote: gradient


function fit!(learner::Learner, phases::AbstractVector{<:AbstractFittingPhase})::Learner
    for phase in phases
        learner.phase = phase
        fitepoch!(learner, phase)
    end
    return learner
end
fit!(learner::Learner, phase::AbstractFittingPhase)::Learner = fit!(learner, [phase])
fit!(learner, n::Int)::Learner = fit!(learner, repeat([TrainingPhase(), ValidationPhase()], n))

"""
    fitepoch!(learner, phase)

Fit `learner` for one epoch.

Customize by deriving custom phase from `AbstractFittingPhase`.
"""
function fitepoch!(
        learner::Learner,
        phase::Union{TrainingPhase, ValidationPhase, TestPhase},
        cbhandler::CallbackHandler = CallbackHandler(learner))
    @assert learner.phase == phase

    EpochBegin() |> cbhandler

    for batch in getdataloader(learner.databunch, phase)
        fitbatch!(learner, phase, batch, cbhandler)
    end

    EpochEnd() |> cbhandler
end


"""
    fitbatch!(learner, phase, batch, cbhandler)

Fit `learner` for one batch.

Customize by deriving custom phase from `AbstractFittingPhase`.
"""
function fitbatch!(
        learner::Learner,
        phase::AbstractTrainingPhase,
        batch,
        cbhandler::CallbackHandler = CallbackHandler(learner))

    BatchBegin() |> cbhandler
    learner.batch = BatchState()
    learner.batch.batch = x, y = learner.device(batch)

    gs = learner.batch.gradients = gradient(learner.params) do
        y_pred = learner.batch.y_pred = learner.model(x)

        LossBegin() |> cbhandler
        loss = learner.batch.loss = learner.lossfn(y_pred, y)

        BackwardBegin() |> cbhandler
        return learner.batch.loss
    end

    BackwardEnd() |> cbhandler

    update!(learner.opt, learner.params, gs)

    BatchEnd() |> cbhandler
end


function fitbatch!(
        learner::Learner,
        phase::ValidationPhase,
        batch,
        cbhandler::CallbackHandler = CallbackHandler(learner))
    BatchBegin() |> cbhandler
    learner.batch = BatchState()
    learner.batch.batch = x, y = learner.device(batch)

    y_pred = learner.batch.y_pred = learner.model(x)

    LossBegin() |> cbhandler
    loss = learner.batch.loss = learner.lossfn(y_pred, y)

    BatchEnd() |> cbhandler
end
