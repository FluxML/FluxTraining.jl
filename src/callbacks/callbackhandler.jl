
struct CallbackHandler
    learner
    callbacks
    CallbackHandler(learner, callbacks) = new(
        learner,
        sort!(callbacks, by=cb -> order(typeof(cb)))
    )
end

function (ch::CallbackHandler)(event::FitEvent)
    foreach(ch.callbacks) do cb
        on(event, ch.learner.phase, cb, ch.learner)
    end
end
