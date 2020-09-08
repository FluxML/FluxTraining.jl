
struct ProtectedException <: Exception end

struct Protected{T, W<:Union{Nothing, Tuple, NamedTuple}}
    data::T
    writable::W
end

function Base.getproperty(protected::Protected, field::Symbol)
    data = getfield(protected, :data)
    writable = getfield(protected, :writable)
    return protectfield(data, field, writable)
end

function protectfield(data, field::Symbol, writable::Tuple)
    fielddata = getproperty(data, field)
    if !(field ∈ writable)
        fielddata = protect(fielddata)
    end
    return fielddata
end

function protectfield(data, field::Symbol, writable::NamedTuple)
    return protect(getproperty(data, field), get(writable, field, nothing))
end

protectfield(data, field::Symbol, writable::Nothing) = protect(getproperty(data, field))


function Base.setproperty!(protected::Protected{T, <:Tuple}, field::Symbol, x) where T
    if field in getfield(protected, :writable)
        setproperty!(getfield(protected, :data), field, x)
    else
        throw(ProtectedException())
    end
end

function Base.setproperty!(protected::Protected, field::Symbol, x) where T
    if allowwrite(getfield(protected, :writable), field)
        setproperty!(getfield(protected, :data), field, x)
    else
        throw(ProtectedException())
        #    "Can't set field $(string(field)) on `Protected` type $(T)!")
    end
end

allowwrite(writable::Tuple, field) = field ∈ writable
allowwrite(writable::NamedTuple, field) = field ∈ keys(writable)
allowwrite(writable::Nothing, field) = false

Base.fieldnames(protected::Protected) = fieldnames(typeof(getfield(protected, :data)))

protect(x, writable) = Protected(x, writable)

protect(x) = protect(x, nothing)

protect(x::Number, _ = nothing) = x
protect(x::AbstractArray, _ = nothing) = x
protect(x::Dict, _ = nothing) = copy(x)
