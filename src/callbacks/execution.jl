

abstract type CallbackExecutor end

struct LinearExecutor <: CallbackExecutor end

function handle(executor::LinearExecutor, event::FitEvent, phase, learner)
    idxs = Zygote.ignore() do
        topologicalsort(learner.callbacks.graph)
    end
    for i in idxs
        _on(event, phase, learner.callbacks.cbs[i], learner)
    end
end


function topologicalsort(g::AbstractGraph)
    nodes = Int[]
    unvisited = trues(nv(g))
    while any(unvisited)
        v = findfirst(unvisited)
        dfs(g, v) do v
            push!(nodes, v)
            unvisited[v] = false
        end
    end
    return nodes
end

function dfs(f, g, v, visited = falses(nv(g)))
    for v_ in inneighbors(g, v)
        if !visited[v_]
            dfs(f, g, v_, visited)
        end
    end
    visited[v] = true
    return f(v)
end
