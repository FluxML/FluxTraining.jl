
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

"""
    PropDict(dict)

Like a `Dict{Symbol}`, but attribute syntax can be used to access values.
"""
struct PropDict{V}
    d::Dict{Symbol, V}
end

Base.getproperty(d::PropDict, field::Symbol) = getfield(d, :d)[field]
Base.setproperty!(d::PropDict, field::Symbol, val) = (getfield(d, :d)[field] = val;)
Base.propertynames(d::PropDict) = Tuple(keys(getfield(d, :d)))

Base.haskey(d::PropDict, key) = haskey(getfield(d, :d), key)
Base.getindex(d::FluxTraining.PropDict{Any}, i::Symbol) = getproperty(d, i)
Base.keys(d::PropDict) = keys(getfield(d, :d))
Base.values(d::PropDict) = values(getfield(d, :d))
Base.get(d::PropDict, args...; kwargs...) = get(getfield(d, :d), args...; kwargs...)

PropDict(args...) = PropDict(Dict{Symbol, Any}(args...))


# ## Tests


@testset "`protect`" begin
    mutable struct C
        x::Int
        y::Int
    end

    mutable struct B
        c::C
    end

    mutable struct A
        b1::B
        b2::B
    end


    makea() = A(
        B(C(1, 2)),
        B(C(3, 4)),
    )

    @testset "fully protected" begin
        a_p = protect(makea(), NamedTuple())  # protect all child structs of `a`
        @test_throws ProtectedException a_p.b1 isa Protected
    end

    @testset "read direct child" begin
        a_p2 = protect(makea(), (b1=Read(),))  # protect everything but `b1`
        @test_nowarn a_p2.b1
        @test_throws ProtectedException a_p2.b1 = C(5, 5)
    end

    @testset "write direct child" begin
        a_p2 = protect(makea(), (b1=Write(),))  # protect everything but `b1`
        @test_nowarn a_p2.b1
        @test_nowarn a_p2.b1 = B(C(5, 5))
    end
    @testset "write nested" begin
        a_p3 = protect(makea(), (b1 = (c = (x=Write(),),),))  # allow mutating only a.b1.c.x
        @test_nowarn a_p3.b1.c.x = 2
        @test_nowarn a_p3.b1.c.x
        @test a_p3.b1.c isa Protected
        @test_throws ProtectedException a_p3.b1.c.y = 2
    end
end
