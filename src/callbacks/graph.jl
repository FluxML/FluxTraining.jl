
# TOOD: check for reads on :cbstate without a previous write (missing callback)
# TODO: check for cyclical dependencies
"""
    callbackgraph(callbacks) -> SimpleDiGraph

Creates a directed acyclic graph from a list of `callbacks`.
Ordering is given through `runafter` and `resolveconflict`.

If a write conflict cannot be resolved (i.e. `resolveconflict`)
is not implemented), throws an error.
"""
function callbackgraph(callbacks)
    # create a graph
    g = SimpleDiGraph(length(callbacks))

    # add dependencies defined through `runafter`
    foreach(e -> add_edge!(g, e), edgesrunafter(callbacks))


    permissions = stateaccess.(callbacks)
    writeaccesses = [accesses(ps, Write) for ps in permissions]
    readaccesses = [accesses(ps, Read) for ps in permissions]

    # detect write-write conflicts and handle them
    for (i, j, accessi, accessj) in findconflicts(writeaccesses, writeaccesses)
        if !(has_edge(g, i, j) || has_edge(g, j, i))
            cb1, cb2 = callbacks[i], callbacks[j]
            resolution = resolveconflict(cb1, cb2)
            if resolution isa NotDefined
                errorwriteconflict(cb1, cb2, accessi, accessj, resolvable = true)
            end
            if resolution isa Unresolvable
                errorwriteconflict(cb1, cb2, accessi, accessj, resolvable = false)
            elseif resolution isa RunFirst
                if resolution.cb === cb1
                    add_edge!(g, cb1, cb2)
                else
                    add_edge!(g, cb2, cb1)
                end
            # resolution isa NoConflict
            else
                continue
            end
        end
    end

    # detect read-write conflicts and handle them
    # writes will be places before reads!
    for (i, j, access) in findconflicts(writeaccesses, readaccesses)
        add_edge!(g, i, j)
    end

    # TODO: check if callback state is read without being written

    # TODO: check for cyclical dependencies
    return g
end

function findconflicts(accesses1, accesses2)
    conflicts = []
    for (i, as1) in enumerate(accesses1)
        for (j, as2) in enumerate(accesses2[i + 1:end])
            for a1 in as1
                for a2 in as2
                    if hasconflict(a1, a2)
                        push!(conflicts, (i, j+i, a1, a2))
                    end
                end
            end
        end
    end
    return conflicts
end


"""
    edgesrunafter(callbacks)

Return a vector of `Edge`s representing dependencies
defined by `runafter`.
"""
function edgesrunafter(callbacks)
    edges = Edge{Int}[]
    for (i, cb) in enumerate(callbacks)
        Ts = runafter(cb)
        for (j, othercb) in enumerate(callbacks)
            if any([othercb isa T for T in  Ts])
                push!(edges, Edge(j, i))
            end
        end
    end
    return edges
end




"""
    accesses()

Enumerate all valid state accesses of permissions of kind `perm`.

`accesses((x = Read(),), Read()) === [(:x,)]`
`accesses((x = Read(),), Write()) === []`

"""
function accesses(permissions::NamedTuple, P::Type{<:Permission} = Permission, a = [], prefix = ())
    for (field, perm) in zip(keys(permissions), permissions)
        if perm isa P
            push!(a, (prefix..., field))
        elseif perm isa NamedTuple
            accesses(perm, P, a, (prefix..., field))
        end
    end
    a
end


function hasconflict(access1::T1, access2::T2)::Bool where {T1<:Tuple, T2<:Tuple}
    l1, l2 = length(access1), length(access2)
    l1 > 0 || return false
    l2 > 0 || return false
    if access1[1] != access2[1]
        return false
    elseif (l1 == 1 || l2 == 1)
        return true
    else
        return hasconflict(access1[2:end], access2[2:end])
    end
end



function errorwriteconflict(cb1, cb2, access1, access2; resolvable = true)
    msg = """
    Write conflict detected between $cb1 and $(cb2)!

    - $cb1 writes to $(formataccess(access1))
    - $cb2 writes to $(formataccess(access2))
    """

    if resolvable
        msg *= """

        To resolve this, implement:

        `resolveconflict(::$(typeof(cb1)), ::$(typeof(cb2)))`

        See also `resolveconflict` and `ConflictResolution`.
        """
    else
        msg *= """

        Both callbacks cannot be used together, please remove one.
        """
    end

    error(msg)
end

formataccess(access) = join(string.(access), '.')
