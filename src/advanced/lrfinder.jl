@with_kw struct LRFinderPhase <: AbstractTrainingPhase
    startlr::Float32 = 1e-7
    endlr::Float32 = 10
    nsteps::Int = 100
    stop_on_divergence::Bool = true
end


function FluxTraining.fitepochphase!(
        learner::Learner,
        phase::LRFinderPhase,
        cbhandler::FluxTraining.CallbackHandler = FluxTraining.CallbackHandler(learner))

    # create Scheduler and Loss callbacks
    epochlength = length(learner.databunch.traindl)
    nepochs = min(1, phase.nsteps / epochlength)
    schedule = Dict(FluxTraining.LR => [ParamSchedule(nepochs, phase.startlr, phase.endlr, anneal_exp)])

    recorder = Recorder()
    callbacks = [ParamScheduler(schedule), recorder]
    cbhandler = FluxTraining.CallbackHandler(learner, callbacks)

    stepstaken = 0
    best_loss = Inf64
    losses = Float64[]
    p = Progress(phase.nsteps; desc = "Learning rate finder: ")

    while stepstaken < phase.nsteps
        for batch in FluxTraining.getdataloader(learner.databunch, phase)
            FluxTraining.fitbatch!(learner, batch, phase, cbhandler)

            loss = learner.batch.loss
            push!(losses, loss)
            if phase.stop_on_divergence
                best_loss = min(loss, best_loss)

                # stop if loss is diverging
                if isnan(loss) || loss > 4best_loss
                    stepstaken = phase.nsteps
                    break
                end
            end

            next!(p)
            stepstaken += 1
            if stepstaken > phase.nsteps
                break
            end
        end
    end

    return losses
    end
