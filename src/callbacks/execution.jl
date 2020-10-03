

abstract type CallbackExecutor end

struct LinearExecutor <: CallbackExecutor end

function handle(executor::LinearExecutor, event::FitEvent, phase, learner)
    idxs = Zygote.ignore() do
        topological_sort_by_dfs(learner.callbacks.graph)
    end
    for i in idxs
        _on(event, phase, learner.callbacks.cbs[i], learner)
    end
end
