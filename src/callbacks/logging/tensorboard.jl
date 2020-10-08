
"""
    TensorBoardBackend(logdir[, tb_overwrite];
        time=time(),
        purge_step=nothing,
        step_increment=1,
        min_level=Logging.Info)

TensorBoard backend for [`Logger`](#). Takes the same arguments
as `TensorBoardLogger.TBLogger`.
"""
struct TensorBoardBackend <: LoggerBackend
    logger::TBLogger
    function TensorBoardBackend(
            logdir,
            existfn = TensorBoardLogger.tb_overwrite; kwargs...)
        return new(TBLogger(logdir, existfn; kwargs...))
    end
end

Base.show(io::IO, backend::TensorBoardBackend) = print(
    io, "TensorBoardBackend(", backend.logger.logdir, ")")

canlog(::TensorBoardBackend) = (Text, Image, Value)


function log_to(backend::TensorBoardBackend, value::Loggables.Value, name, i; group = ())
    name = _combinename(name, group)
    log_value(backend.logger, name, value.data; step = i)
end


function log_to(backend::TensorBoardBackend, image::Loggables.Image, name, i; group = ())
    name = _combinename(name, group)
    log_image(backend.logger, name, image.data; step = i)
end


function log_to(backend::TensorBoardBackend, text::Loggables.Text, name, i; group = ())
    name = _combinename(name, group)
    log_text(backend.logger, name, text.data; step = i)
end





# Utilities

_combinename(name, group::String) = _combinename((group, name))
_combinename(name, group::Tuple) = _combinename((group..., name))
_combinename(strings::Tuple) = join(strings, '/')
