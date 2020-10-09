"""
    module Loggables

Defines [`Loggables.Loggable`](#) and its subtypes.
"""
module Loggables

"""
    abstract type Loggable

Abstract type for data that [`LoggerBackend`](#)s can log.
See `subtypes(FluxTraining.Loggables.Loggable)` and [`LoggerBackend`](#)

"""
abstract type Loggable end

struct Text <: Loggable data end
struct Image <: Loggable data end
struct Audio <: Loggable data end
struct Histogram <: Loggable data end
struct Value <: Loggable data end
struct Graph <: Loggable data end
struct File <: Loggable
    file
    data
end

end  # module
