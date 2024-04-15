using MLFlowLogger: MLFLogger, log_metric

"""
    MLFlowBackend(;
        tracking_uri=nothing, 
        experiment_name=nothing,
        run_id=nothing,
        start_step=0,
        step_increment=1,
        min_level=CoreLogging.Info,
        kwargs...)
MLFlow backend for logging callbacks. Takes the same arguments
as [`MLFlowLogger.MLFlowLogger`](https://github.com/rejuvyesh/MLFlowLogger.jl/blob/master/src/MLFlowLogger.jl).
"""
struct MLFlowBackend <: LoggerBackend
    logger::MLFLogger
    
    function MLFlowBackend(; kwargs...)
        return new(MLFLogger(; kwargs...))
    end
end

Base.show(io::IO, backend::MLFlowBackend) = print(
    io, "MLFlowBackend(", backend.logger, ")")

function log_to(backend::MLFlowBackend, value::Loggables.Value, name, i; group = ())
    name = _combinename(name, group)
    log_metric(backend.logger, name, value.data; step = i)
end
