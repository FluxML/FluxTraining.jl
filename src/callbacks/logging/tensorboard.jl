
"""
    TensorBoardBackend(logdir[, tb_overwrite];
        time=time(),
        purge_step=nothing,
        step_increment=1,
        min_level=Logging.Info)

TensorBoard backend for logging callbacks. Takes the same arguments
as [`TensorBoardLogger.TBLogger`](https://julialogging.github.io/TensorBoardLogger.jl/dev/).
"""
struct TensorBoardBackend <: LoggerBackend
    logger::TBLogger
    function TensorBoardBackend(
            logdir,
            existfn = tb_overwrite; kwargs...)
        return new(TBLogger(logdir, existfn; kwargs...))
    end
end

Base.show(io::IO, backend::TensorBoardBackend) = print(
    io, "TensorBoardBackend(", backend.logger.logdir, ")")


function log_to(backend::TensorBoardBackend, value::Loggables.Value, name, i; group = ())
    name = _combinename(name, group)
    log_value(backend.logger, name, value.data; step = i)
end


function log_to(backend::TensorBoardBackend, image::Loggables.Image, name, i; group = ())
    name = _combinename(name, group)
    im = ImageCore.clamp01nan!(collect(image.data))
    log_image(backend.logger, name, im; step = i)
end


function log_to(backend::TensorBoardBackend, text::Loggables.Text, name, i; group = ())
    name = _combinename(name, group)
    log_text(backend.logger, name, text.data; step = i)
end

function log_to(backend::TensorBoardBackend, hist::Loggables.Histogram, name, i; group=())
    name = _combinename(name, group)
    log_histogram(backend.logger, name, hist.data, step=i)
end
