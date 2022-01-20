mutable struct TimingRecorder <: Callback
    forward_pass_start_time::Float64
    backward_pass_start_time::Float64
    optimizer_start_time::Float64
    validation_start_time::Float64

    forward_pass_total_time::Float64
    backward_pass_total_time::Float64
    optimizer_total_time::Float64
    training_total_time::Float64

    validation_total_time::Float64

    TimingRecorder() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

runafter(::TimingRecorder) = (Metrics,)

stateaccess(::TimingRecorder) = (cbstate = (metricsepoch = Write(), history = Read()),)

Zygote.@nograd time

function on(::StepBegin, phase::AbstractTrainingPhase, cb::TimingRecorder, learner)
    cb.forward_pass_start_time = time()
end

function on(::BackwardBegin, phase::AbstractTrainingPhase, cb::TimingRecorder, learner)
    cb.forward_pass_total_time = time() - cb.forward_pass_start_time
    cb.backward_pass_start_time = time()
end

function on(::BackwardEnd, phase::AbstractTrainingPhase, cb::TimingRecorder, learner)
    cb.backward_pass_total_time = time() - cb.backward_pass_start_time
    cb.optimizer_start_time = time()
end

function on(::StepEnd, phase::AbstractTrainingPhase, cb::TimingRecorder, learner)
    cb.optimizer_total_time = time() - cb.optimizer_start_time
    cb.training_total_time = cb.forward_pass_total_time + cb.backward_pass_total_time + cb.optimizer_total_time
end

function on(::StepBegin, phase::AbstractValidationPhase, cb::TimingRecorder, learner)
    cb.validation_start_time = time()
end

function on(::StepEnd, phase::AbstractValidationPhase, cb::TimingRecorder, learner)
    cb.validation_total_time = time() - cb.validation_start_time
end

function on(::EpochEnd, phase::AbstractTrainingPhase, cb::TimingRecorder, learner)
    metricsepoch = learner.cbstate.metricsepoch[phase]
    epoch = learner.cbstate.history[phase].epochs

    push!(metricsepoch, Symbol("Forward Pass Total Time (in s)"), epoch, cb.forward_pass_total_time)
    push!(metricsepoch, Symbol("Backward Pass Total Time (in s)"), epoch, cb.backward_pass_total_time)
    push!(metricsepoch, Symbol("Optimizer Total Time (in s)"), epoch, cb.optimizer_total_time)
    push!(metricsepoch, Symbol("Training Total Time (in s)"), epoch, cb.training_total_time)
end

function on(::EpochEnd, phase::AbstractValidationPhase, cb::TimingRecorder, learner)
    metricsepoch = learner.cbstate.metricsepoch[phase]
    epoch = learner.cbstate.history[phase].epochs

    push!(metricsepoch, Symbol("Validation Total Time (in s)"), epoch, cb.validation_total_time)
end
