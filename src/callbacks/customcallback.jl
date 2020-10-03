
mutable struct CustomCallback{E<:FitEvent,P<:Phase} <: AbstractCallback
    fn::Any
    everyn::Int
    _current::Int
    CustomCallback{E,P}(fn, everyn = 1) where {E,P} = new{E,P}(fn, everyn, 0)
end


function on(
    ::E,
    ::P,
    cb::CustomCallback{E,P},
    learner,
) where {E<:FitEvent,P<:Phase}
    cb._current += 1
    if cb._current % cb.everyn == 0
        cb.fn(learner)
    end
end
