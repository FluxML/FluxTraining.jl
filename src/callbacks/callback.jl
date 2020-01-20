abstract type AbstractCallback end
abstract type AbstractMetric <: AbstractCallback end

order(c::Type{<:AbstractCallback}) = 0
order(c::T) where T<:AbstractCallback = order(T)
