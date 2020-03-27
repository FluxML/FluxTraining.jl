using PyCall

# TODO: factor out into separate package

mlflow = pyimport("mlflow")

struct ExperimentTracker <: AbstractCallback
    experiment::AbstractString
    trackinguri::AbstractString
    parameters::Dict
end

order(::Type{ExperimentTracker}) = 100


function on(::Initialize, ::InitializationPhase, experiment::ExperimentTracker, learner)
    mlflow = pyimport("mlflow")
    mlflow.set_tracking_uri(experiment.trackinguri)
    mlflow.set_experiment(experiment.experiment)
    mlflow.log_param("training_nepochs", "0")
    mlflow.log_param("training_nsteps", "0")

    for (key, value) in experiment.parameters
        mlflow.log_param(string(key), string(value))
    end
end

function on(::EpochEnd, ::AbstractFittingPhase, experiment::ExperimentTracker, learner)
    mlflow = pyimport("mlflow")
    # update params
    mlflow.log_param("training_nepochs", string(learner.recorder.epoch))

    # log metrics
    for metric in learner.metrics
        mlflow.log_metric(string(metric), value(metric), learner.recorder.steptotal)
    end

    # log artifacts
    mlflow.log_artifacts(artifactpath(learner))
end

function on(::BatchEnd, ::AbstractTrainingPhase, experiment::ExperimentTracker, learner)
    mlflow = pyimport("mlflow")
    steps = learner.recorder.steptotal
    mlflow.log_param("training_nsteps", steps)
    mlflow.log_metric("lr", getoptimparam(learner.opt, LR), steps)
    mlflow.log_metric("step_loss", learner.batch.loss, steps)
end


function on(::Cleanup, ::CleanupPhase, experiment::ExperimentTracker, learner)
    mlflow = pyimport("mlflow")
    mlflow.end_run()
end
