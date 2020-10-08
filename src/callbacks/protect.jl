
struct ProtectedException <: Exception
    msg::String
end


abstract type Permission end
struct Read <: Permission end
struct Write <: Permission end

struct Protected{T}
    data::T
    perms::NamedTuple
end


function Base.getproperty(protected::Protected, field::Symbol)
    return getfieldperm(
        getfield(protected, :data),
        field,
        get(getfield(protected, :perms), field, nothing),
    )
end

function Base.setproperty!(protected::Protected, field::Symbol, x)
    return setfieldperm!(
        getfield(protected, :data),
        field,
        x,
        get(getfield(protected, :perms), field, nothing),
    )
end

function Base.getindex(protected::Protected, idx)
    return getindexperm(
        getfield(protected, :data),
        idx,
        get(getfield(protected, :perms), idx, nothing),
    )
end

function Base.setindex!(protected::Protected, x, idx)
    return setindexperm!(
        getfield(protected, :data),
        x,
        idx,
        get(getfield(protected, :perms), idx, nothing),
    )
end

Base.haskey(d::Protected, key) = haskey(getfield(d, :data), key)

getindexperm(data::D, idx, perm::Nothing) where D =
    throw(ProtectedException("Read access to $D[$idx] disallowed."))
getindexperm(data, idx, perm::Union{Read, Write}) = getindex(data, idx)
getindexperm(data, idx, perm::NamedTuple) = protect(getindex(data, idx), perm)

setindexperm!(data::D, idx, x, perm::Union{Read, Nothing, NamedTuple}) where D =
    throw(ProtectedException("Write access to $D[$idx] disallowed."))
setindexperm!(data, x, idx, perm::Write) = setindex!(data, x, idx)


getfieldperm(data::D, field, perm::Nothing) where D =
    throw(ProtectedException("Read access to $(D).$(string(field)) disallowed."))
getfieldperm(data, field, perm::Union{Read, Write}) = getproperty(data, field)
getfieldperm(data, field, perm::NamedTuple) = protect(getproperty(data, field), perm)

setfieldperm!(data::D, field, x, perm::Union{Read, Nothing, NamedTuple}) where D =
    throw(ProtectedException("Write access to $(D).$(string(field)) disallowed."))
setfieldperm!(data, field, x, perm::Write) = setproperty!(data, field, x)


Base.fieldnames(protected::Protected) = fieldnames(typeof(getfield(protected, :data)))

protect(x, perms) = Protected(x, perms)

struct PropDict{V}
    d::Dict
    function PropDict(d::Dict{K, V}) where {K, V}
        return new{V}(d)
    end
end

Base.getproperty(d::PropDict, field::Symbol) = getfield(d, :d)[field]
Base.setproperty!(d::PropDict, field::Symbol, val) = (getfield(d, :d)[field] = val;)
Base.propertynames(d::PropDict) = Tuple(keys(getfield(d, :d)))

Base.haskey(d::PropDict, key) = haskey(getfield(d, :d), key)

PropDict(args...) = PropDict(Dict{Symbol, Any}(args...))
