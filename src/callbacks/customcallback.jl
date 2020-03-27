
struct CustomCallback{E<:FitEvent, P<:AbstractFittingPhase} <: AbstractCallback
    fn
end

function on(::E, ::P, cb::CustomCallback{E, P}, learner) where {E<:FitEvent, P<:AbstractFittingPhase}
    cb.fn(learner)
end
