

abstract type CallbackRunner end

struct LinearRunner <: CallbackRunner end

function handle(runner::LinearRunner, event::Event, phase, learner)
    idxs = Zygote.ignore() do
        topological_sort_by_dfs(learner.callbacks.graph)
    end
    for i in idxs
        _on(event, phase, learner.callbacks.cbs[i], learner)
    end
end
