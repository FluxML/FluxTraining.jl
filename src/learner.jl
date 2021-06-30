

mutable struct Callbacks
    cbs::Vector
    runner::CallbackRunner
    graph::SimpleDiGraph
    initialized::Bool
end
Callbacks(cbs, runner=LinearRunner()) = Callbacks(cbs, runner, callbackgraph(cbs), false)

init!(cbs::Callbacks, learner) = foreach(cb -> init!(cb, learner), cbs.cbs)


mutable struct Learner
    model
    data::PropDict
    optimizer
    lossfn
    params
    step::PropDict
    callbacks::Callbacks
    cbstate::PropDict
end


"""
    Learner(model, data, optimizer, lossfn, [callbacks...; kwargs...])

Holds and coordinates all state of the training. `model` is trained by
optimizing `lossfn` with `optimizer` on `data`.

## Arguments

- `model`: A Flux.jl model or a `NamedTuple` of models.
- `data`: Tuple of data iterators in the order `(traindata, valdata)`.
    A data iterator is an iterable over batches. For regular supervised training,
    each batch should be a tuple `(xs, ys)`.
- `lossfn`: Function with signature `lossfn(model(x), y) -> Number`
- `optimizer`
- `callbacks...`: Any other unnamed arguments are callbacks

## Keyword arguments

- `usedefaultcallbacks = true`: Whether to add some basic callbacks. Included
    are [`Metrics`](#), [`Recorder`](#), [`ProgressPrinter`](#),
    [`StopOnNaNLoss`](#), and [`MetricsPrinter`](#).
- `cbrunner = LinearRunner()`: Callback runner to use.

## Fields

*(Use this as a reference when implementing callbacks)*

- `model`, `optimizer`, and `lossfn` are stored as passed in
- `data` is a `NamedTuple` of `(training = ..., validation = ..., test = ...)`.
    Some values might be `nothing` if you didn't pass in multiple data iterators.
- `params`: An instance of `model`'s parameters of type `Flux.Params`. If `model` is
    a `NamedTuple`, then `params` is a `NamedTuple` as well.
- `step::`[`PropDict`](#): State of the last step. Contents depend on the last run
    [`Phase`](#).
- `cbstate::`[`PropDict`](#): Special state container that callbacks can
    save state to for other callbacks. Its keys depend on what callbacks
    are being used. See the [custom callbacks guide](../docs/callbacks/custom.md)
    for more info.
"""
function Learner(
        model, data, optimizer, lossfn, callbacks::Vararg{<:Callback};
        usedefaultcallbacks=true, cbrunner=LinearRunner()
    )
    callbacks = collect(Callback, callbacks)

    if usedefaultcallbacks
        for cb in defaultcallbacks()
            if !any(typeof(cb) .== typeof.(callbacks))
                push!(callbacks, cb)
            end
        end
    end

    cbs = Callbacks(callbacks, cbrunner)

    learner = Learner(
        model,
        _dataiters(data),
        optimizer,
        lossfn,
        paramsrec(model),
        PropDict(),
        cbs,
        PropDict())
    init!(cbs, learner)
    return learner
end


Base.show(io::IO, learner::Learner) = print(io, "Learner()")



defaultcallbacks()::Vector{AbstractCallback} = [
    ProgressPrinter(),
    MetricsPrinter(),
    StopOnNaNLoss(),
    Recorder(),
    Metrics(),
    SanityCheck(),
]


#  Callback handling
handle(event, learner, phase) = handle(learner.callbacks.runner, event, phase, learner)


# Other

phasedataiter(::AbstractTrainingPhase) = :training
phasedataiter(::AbstractValidationPhase) = :validation


function model!(learner, model)
    learner.model = model
    learner.params = Flux.paramsrec(model)
end


numsteps(learner::Protected, phase) = numsteps(getfield(learner, :data), phase)
numsteps(learner, phase) = length(learner.data[phasedataiter(phase)])

_dataiters(d::PropDict) = d
_dataiters(t::NamedTuple) = PropDict(pairs(t))
function _dataiters(t::Tuple)
    if length(t) == 0
        return PropDict(Dict{Symbol,Any}())
    elseif length(t) == 1
        return _dataiters((training = t[1]))
    elseif length(t) == 2
        return _dataiters((training = t[1], validation = t[2]))
    else
        error("Please pass a `NamedTuple` or `PropDict` as `data`.")
    end
end


paramsrec(m) = params(m)
paramsrec(t::Union{Tuple,NamedTuple}) = map(paramsrec, t)

# Callback utilities
