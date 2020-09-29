


# TODO: handle Write-Read conflicts by putting Write before Read
"""
    callbackgraph(callbacks) -> SimpleDiGraph

Creates a directed acyclic graph from a list of `callbacks`.
Ordering is given through `runafter` and `resolveconflict`.

If a write conflict cannot be resolved (i.e. `resolveconflict`)
is not implemented), throws an error.
"""
function callbackgraph(callbacks)
    vdict = Dict(typeof(cb) => i for (i, cb) in enumerate(cbs))
    length(vdict) == length(callbacks) || error("Cannot handle the same callback passed multiple times")

    g = SimpleDiGraph(length(callbacks))

    # create edges from `runafter`
    for callback in callbacks
        for Callback in runafter(callback)
            add_edge!(g, vdict[Callback], typeof(callback))
        end
    end

    # check for write conflicts and resolve them where possible
    for (v1, v2) in findconflicts(stateaccess.(callbacks))
        # only resolve if an ordering isn't given through `runafter`
        if !(hasedge(g, v1, v2) || hasedge(g, v2, v1))
            cb1, cb2 = callbacks[v1], callbacks[v2]
            resolution = resolveconflict(cb1, cb2)
            # no valid ordering exists, error
            if resolution isa Unresolvable
                C1, C2 = typeof(callbacks[v1]), typeof(callbacks[v2])
                error("Write conflict detected between $C1 and $C2. Implement `resolveconflict(::$(C1), ::$(C2))`")
            # a valid ordering exists, add an edge
            elseif resolution isa RunFirst
                if resolution.cb === cb1
                    add_edge!(g, cb1, cb2)
                else
                    add_edge!(g, cb2, cb1)
                end
            end
            # if `resolution isa NoConflict`, no edge needs to be added
        end
    end

    return g
end


"""
    findconflicts(accesses)

Finds write conflicts between any pair of `accesses`.

`findconflicts([(x = Write(),), (x = Write(),)] == [(1, 2)]`
`findconflicts([(x = Write(),), (y = Write(),)] == []`
"""
function findconflicts(accesses)
    writables = writable.(accesses)
    conflicts = Tuple{Int, Int}[]
    for (i, ws1) in enumerate(writables)
        for (j, ws2) in enumerate(writables[i+1:end])
            if any([hasconflict(w1, w2) for (w1, w2) in Iterators.product(ws1, ws2)])
                push!(conflicts, (i, j+1))
            end
        end
    end
    return conflicts
end


function writable(permission::NamedTuple, a = [], prefix = ())
    for field in keys(permission), perm in permission
        if perm isa Write
            push!(a, (prefix..., field))
        elseif perm isa NamedTuple
            writable(perm, a, (prefix..., field))
        end
    end
    a
end


function hasconflict(write1::T1, write2::T2) where {T1<:Tuple, T2<:Tuple}
    l1, l2 = length(write1), length(write2)
    l1 > 0 || return false
    l2 > 0 || return false
    if write1[1] != write2[1]
        return false
    elseif (l1 == 1 || l2 == 1)
        return true
    else
        return hasconflict(write1[2:end], write2[2:end])
    end
end
