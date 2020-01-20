using OnlineStats: Mean, ExponentialWeight

struct LRFinderPhase <: AbstractTrainingPhase
    start_lr::Real
    end_lr::Real
    nbatches::Integer
end
# TODO: refactor for new callbacks

LRFinderPhase(start_lr = 1e-7, end_lr = 10; nbatches = 100) = LRFinderPhase(
    start_lr, end_lr, nbatches)

function fitepoch!(learner::Learner, phase::LRFinderPhase)
    learner.phase = phase
    epochs = phase.nbatches / numsteps(learner, phase)

    schedule = Dict(LR => [ParamSchedule(epochs, phase.start_lr, phase.end_lr, anneal_exp)])

    setschedule!(learner, schedule)
    smoothloss = MeanMetric(learner.lossfn, ExponentialWeight(0.02))
    push!(learner.metrics, smoothloss)
    cbhandler = CallbackHandler(learner)

    EpochBegin() |> cbhandler

    minloss = typemax(Float32)
    for (b, batch) in enumerate(learner.databunch.traindl)
        BatchBegin() |> cbhandler

        fitbatch!(learner, phase, batch, cbhandler)
        loss = value(smoothloss)

        BatchEnd() |> cbhandler
        if loss < minloss
            minloss = loss
        end
        if isnan(learner.batch.loss) || (4minloss < loss)
            break
        end
        if b == phase.nbatches
            break
        end
    end

    EpochBegin() |> cbhandler

    return learner
end
