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
    # events
    # events should include time, phase, event, endtime
    events::Vector = []
end

"""
    Recorder(epoch, step, epochstats, stepstats)
"""
struct Recorder <: AbstractCallback end


order(cb::Type{Recorder}) = -90


function on(::EpochBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    learner.state.history.nstepsepoch = 0
end


function on(::BatchEnd, phase::AbstractTrainingPhase, recorder::Recorder, learner)
    history = learner.state.history

    # increment counters
    history.nsteps += 1
    history.nstepsepoch += 1
    history.nsamples += size(learner.state.batch.xs)[end]

    # save metrics
    push!(history.stepmetrics, :loss, history.nsteps, learner.state.batch.loss)
    for metric in learner.state.callbacks.metrics
        push!(history.stepmetrics, Symbol(metric), history.nsteps, metric.last)
    end

    # save hyper parameters
    for (OptimParam, value) in getoptimparams(learner.opt)
        push!(history.stephyperparams, Symbol(OptimParam.name), history.nsteps, value)
    end
end


function on(::EpochEnd, phase::AbstractFittingPhase, recorder::Recorder, learner)
    history = learner.state.history
    cbs = learner.state.callbacks
    if phase isa AbstractTrainingPhase
        history.epochs += 1
    end

    push!(history.epochmetrics[phase], :loss, history.epochs, value(cbs.loss))
    for metric in cbs.metrics
        push!(history.epochmetrics[phase], string(metric), history.epochs, value(metric))
    end
end
