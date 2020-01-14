import Flux: params, gpu
import Flux.Optimise: update!
import Zygote: Params, gradient
import IterTools: imap


function fit!(learner::Learner, phases::AbstractVector{AbstractFittingPhase})
    for phase in phases
        fitepoch!(learner, phase)
    end
end


"""
    fitepoch!(learner, phase)

Fit `learner` for one epoch.

Customize by deriving custom phase from `AbstractFittingPhase`.
"""
function fitepoch!(
        learner::Learner,
        phase::Union{TrainingPhase, ValidationPhase, TestPhase})

    cbhandler = CallbackHandler(
        [learner.metrics..., learner.recorder, learner.callbacks...], learner.state)

    on_epoch_begin(cbhandler, phase)
    for batch in getdataloader(learner.databunch, phase)
        fitbatch!(learner, phase, batch, cbhandler)
    end
    on_epoch_end(cbhandler)
end


"""
    fitbatch!(learner, phase, batch, cbhandler)

Fit `learner` for one batch.

Customize by deriving custom phase from `AbstractFittingPhase`.
"""
function fitbatch!(learner::Learner, phase::AbstractTrainingPhase, batch, cbhandler::CallbackHandler)
    ps = cbhandler.state.params
    batch = on_batch_begin(cbhandler, batch)
    x, y = batch |> learner.device
    gs = gradient(ps) do
        output = learner.model(x)
        on_loss_begin(cbhandler, output)
        lossbatch = learner.lossfn(output, y)
        on_backward_begin(cbhandler, lossbatch)
        return lossbatch
    end
    on_backward_end(cbhandler, gs)
    update!(learner.opt, ps, gs)
    on_batch_end(cbhandler)
end


function fitbatch!(learner::Learner, phase::ValidationPhase, batch, cbhandler::CallbackHandler)
    batch = on_batch_begin(cbhandler, batch)
    x, y = batch |> learner.device
    output = learner.model(x)
    on_loss_begin(cbhandler, output)
    lossbatch = learner.lossfn(output, y)
    on_backward_begin(cbhandler, lossbatch)
    on_batch_end(cbhandler)
end
