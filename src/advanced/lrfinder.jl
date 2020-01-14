using OnlineStats: Mean, ExponentialWeight

struct LRFinderPhase <: AbstractTrainingPhase
    start_lr::Real
    end_lr::Real
    nbatches::Integer
    stopondiv::Bool
end

LRFinderPhase(start_lr = 1e-7, end_lr = 10; nbatches = 100, stopondiv = true) = LRFinderPhase(
    start_lr, end_lr, nbatches, stopondiv)

function fitepoch!(learner::Learner, phase::LRFinderPhase)
    epochs = phase.nbatches / length(getdataloader(learner.databunch, phase))

    paramscheduler = ParamScheduler(
        Dict(LR => [ParamSchedule(epochs, phase.start_lr, phase.end_lr, anneal_exp)])
    )
    learner.state.epoch += 1
    recorder = Recorder()
    smoothloss = Mean(Float32, weight = ExponentialWeight(0.02))
    cbhandler = CallbackHandler([recorder, paramscheduler], learner.state)

    on_epoch_begin(cbhandler, phase)
    minloss = typemax(Float32)
    for (b, batch) in enumerate(learner.databunch.traindl)
        fitbatch!(learner, phase, batch, cbhandler)
        learner.state.step += 1
        OnlineStats.fit!(smoothloss, learner.state.lossbatch)
        l = OnlineStats.value(smoothloss)
        if l < minloss
            minloss = l
        end
        if isnan(learner.state.lossbatch) || (4minloss < learner.state.lossbatch)
            break
        end
        if b == phase.nbatches
            break
        end
    end

    return recorder
end
