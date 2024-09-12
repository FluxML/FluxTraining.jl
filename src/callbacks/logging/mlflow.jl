using MLFlowClient

"""
    MLFlowBackend(tracking_uri, experiment_name[; kwargs...])

MLFlow backend for logging metrics. Creates a new experiment if it doesn't exist
and starts a new run.
"""
struct MLFlowBackend <: LoggerBackend
    mlf::MLFlowClient.MLFlow
    experiment::MLFlowClient.MLFlowExperiment
    run::MLFlowClient.MLFlowRun
    function MLFlowBackend(tracking_uri::String, experiment_name::String; kwargs...)
        mlf = MLFlowClient.MLFlow(tracking_uri)
        experiment = MLFlowClient.getorcreateexperiment(mlf, experiment_name)
        run = MLFlowClient.createrun(mlf, experiment)
        return new(mlf, experiment, run)
    end
end

Base.show(io::IO, backend::MLFlowBackend) = print(
    io, "MLFlowBackend(", backend.mlf.apiroot, ", ", backend.experiment.name, ")")

function log_to(backend::MLFlowBackend, value::Loggables.Value, name, step; group = ())
    name = _combinename(name, group)
    MLFlowClient.logmetric(backend.mlf, backend.run, name, value.data, step=step)
end
