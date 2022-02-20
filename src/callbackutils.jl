

"""
    setcallbacks!(learner, callbacks)

Set `learner`'s callbacks to `callbacks`, removing all current callbacks.
"""
function setcallbacks!(learner, callbacks)
    learner.callbacks = Callbacks(callbacks)
end


"""
    addcallback!(learner, callback)

Adds `callback` to `learner` and updates the dependency graph.
"""
function addcallback!(learner, callback::AbstractCallback)
    learner.callbacks = Callbacks(vcat(learner.callbacks.cbs, callback))
    init!(callback, learner)
    return
end


"""
    getcallback(learner, C)

Find callback of type `C` in `learner`'s callbacks and return it.
If there is none, return `nothing`.
"""
function getcallback(learner, C::Type{<:FluxTraining.Callback})
    cbidx = findfirst(isa.(learner.callbacks.cbs, C))
    return isnothing(cbidx) ? nothing : learner.callbacks.cbs[cbidx]
end


"""
    replacecallback!(learner, callback::C)

Replace existing callback of type `C` on learner with `callback`.
Return the replaced callback.

If `learner` doesn't have a callback of type `C`, add `callback` and
return `nothing`.
"""
function replacecallback!(learner, callback::C) where {C <: FluxTraining.Callback}
    cbidx = findfirst(isa.(learner.callbacks.cbs, C))
    if isnothing(cbidx)
        FluxTraining.addcallback!(learner, callback)
        return nothing
    else
        oldcb = learner.callbacks.cbs[cbidx]
        learner.callbacks.cbs[cbidx] = callback
        FluxTraining.setcallbacks!(learner, learner.callbacks.cbs)
        return oldcb
    end
end


"""
    removecallback!(learner, C)

Remove the first callback of type `C` from `learner` and return it.
If there is none, return `nothing`.
"""
function removecallback!(learner, C::Type{<:FluxTraining.Callback})
    cbidx = findfirst(isa.(learner.callbacks.cbs, C))
    if isnothing(cbidx)
        return nothing
    end
    cb = popat!(learner.callbacks.cbs, cbidx)
    learner.callbacks = Callbacks(learner.callbacks.cbs)
    return cb
end
