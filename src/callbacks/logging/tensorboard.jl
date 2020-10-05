
struct TensorBoardBackend <: LoggerBackend
    logger::TBLogger
end
TensorBoardBackend(args...; kwargs...) = TensorBoardBackend(TBLogger(args...; kwargs...))

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

_combinename(name, group::String) = _combinename((name, group))
_combinename(name, group::Tuple) = _combinename((name, group...))
_combinename(strings::Tuple) = join(strings, '/')
