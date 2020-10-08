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
end

"""
    Recorder(epoch, step, epochstats, stepstats)
"""
struct Recorder <: Callback end


stateaccess(::Recorder) = (
    cbstate = (history = Write(),),
    batch = Read(),
    )


function on(::Init, ::Phase, recorder::Recorder, learner)
    learner.cbstate[:history] = History()
end


function on(::EpochBegin, ::AbstractTrainingPhase, recorder::Recorder, learner)
    learner.cbstate[:history].nstepsepoch = 0
end


function on(::BatchEnd, phase::AbstractTrainingPhase, recorder::Recorder, learner)
    history = learner.cbstate[:history]
    history.nsteps += 1
    history.nstepsepoch += 1
    history.nsamples += size(learner.batch.xs)[end]
end


function on(::EpochEnd, phase::Phase, recorder::Recorder, learner)
    history = learner.cbstate[:history]
    if phase isa AbstractTrainingPhase
        history.epochs += 1
    end
end
