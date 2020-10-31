@with_kw mutable struct History
    # Number of completed training epochs
    epochs::Int = 0
    # total number of completed steps
    steps::Int = 0
    # number of completed steps in current epoch
    stepsepoch::Int = 0
    # number of seen samples during training, usually `nsteps * batchsize`
    samples::Int = 0
end

"""
    Recorder()

Maintains a [`History`](#). It's stored in `learner.cbstate.history`.
"""
struct Recorder <: Callback end


stateaccess(::Recorder) = (
    cbstate = (history = Write(),),
    batch = Read(),
    )


function on(::Init, ::Phase, recorder::Recorder, learner)
    learner.cbstate.history = History()
end


function on(::EpochBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    learner.cbstate.history.stepsepoch = 0
end


function on(::BatchEnd, phase::AbstractTrainingPhase, recorder::Recorder, learner)
    history = learner.cbstate.history
    history.steps += 1
    history.stepsepoch += 1
    history.samples += size(learner.batch.xs)[end]
end


function on(::EpochEnd, phase::Phase, recorder::Recorder, learner)
    history = learner.cbstate.history
    if phase isa AbstractTrainingPhase
        history.epochs += 1
    end
end
