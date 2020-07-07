
const FluxOptimizer = Union{
    Descent, Momentum, Nesterov, RMSProp,
    ADAM, AdaMax, ADAGrad, ADADelta, RADAM,
    AMSGrad, NADAM
}


abstract type OptimParam end
struct LR <: OptimParam end
struct Mom <: OptimParam end


optimparamfield(::O, op::Type{P}) where {O, P<:OptimParam} = error(
    "No parameter $(P) on optimizer $(O)."
)
optimparamfield(::FluxOptimizer, ::Type{LR}) = :eta

function setoptimparam!(opt, optimparam::Type{<:OptimParam}, x)
    setfield!(opt, optimparamfield(opt, optimparam), x)
end
getoptimparam(opt, optimparam::Type{<:OptimParam}) = getfield(opt, optimparamfield(opt, optimparam))

# TODO: add momentum, weight decay and other...
getoptimparams(opt) = Dict(LR => getoptimparam(opt, LR))
