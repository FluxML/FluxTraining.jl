"""
    $TYPEDEF

$TYPEDFIELDS
"""
@with_kw mutable struct History
    # Number of completed training epochs
    epochs::Int = 0
    # total number of completed steps
    nsteps::Int = 0
    # number of completed steps in current epoch
    nstepsepoch::Int = 0
    # number of seen samples during training, usually `nsteps * batchsize`
    nsamples::Int = 0
    # stores metrics of epochs grouped by phase
    epochmetrics::DefaultDict =
        DefaultDict{AbstractFittingPhase, MVHistory}(() -> MVHistory())
    # stores values of all metrics for every step
    stepmetrics::MVHistory = MVHistory()
    # stores values of all hyperparameters for every step
    stephyperparams::MVHistory = MVHistory()
end

"""
    Recorder(epoch, step, epochstats, stepstats)
"""
struct Recorder <: Callback end


stateaccess(::Recorder) = (cbstate = (history = Write()), batch = Read(), callbacks = Read())
runafter(::Recorder) = (AbstractMetric,)


function on(::EpochBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    if isnothing(get(learner.cbstate, :history, nothing))
        learner.cbstate[:history] = History()
    end
    learner.cbstate[:history].nstepsepoch = 0
end


function on(::BatchEnd, phase::AbstractTrainingPhase, recorder::Recorder, learner)
    history = learner.cbstate[:history]

    # increment counters
    history.nsteps += 1
    history.nstepsepoch += 1
    history.nsamples += size(learner.batch.xs)[end]

    # save metrics
    for metric in getmetrics(learner.callbacks)
        push!(history.stepmetrics, Symbol(metric), history.nsteps, stepvalue(metric))
    end

    # TODO: refactor together with hyperparameters
    # save hyper parameters
    # for (OptimParam, value) in getoptimparams(learner.opt)
    #     push!(history.stephyperparams, Symbol(OptimParam.name), history.nsteps, value)
    # end
end


function on(::EpochEnd, phase::AbstractFittingPhase, recorder::Recorder, learner)
    history = learner.cbstate[:history]

    if phase isa AbstractTrainingPhase
        history.epochs += 1
    end

    for metric in getmetrics(learner.callbacks)
        push!(history.epochmetrics[phase], Symbol(string(metric)), history.epochs, epochvalue(metric))
    end
end
