

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
    # this used to store `Flux.Params` but now stores the optimiser state
    # if an optim from Optimisers.jl is used
    params
    step::PropDict
    callbacks::Callbacks
    cbstate::PropDict
    metadata::PropDict
end


"""
    Learner(model, lossfn; [callbacks = [], optimizer = ADAM(), kwargs...])

Holds and coordinates all state of the training. `model` is trained by
optimizing `lossfn` with `optimizer` on `data`.

## Arguments

Positional arguments:

- `model`: A Flux.jl model or a `NamedTuple` of models.
- `lossfn`: Loss function with signature `lossfn(model(x), y) -> Number`.

Keyword arguments (optional):

- `data = ()`: Data iterators. A 2-tuple will be treated as `(trainingdataiter, validdataiter)`.
    You can also pass in an empty tuple `()` and use the [`epoch!`](#) method with a
    `dataiter` as third argument.

    A data iterator is an iterable over batches. For regular supervised training,
    each batch should be a tuple `(xs, ys)`.
- `optimizer = ADAM()`: The optimizer used to update the `model`'s weights
- `callbacks = []`: A list of callbacks that should be used. If `usedefaultcallbacks == true`,
    this will be extended by the default callbacks
- `usedefaultcallbacks = true`: Whether to add some basic callbacks. Included
    are [`Metrics`](#), [`Recorder`](#), [`ProgressPrinter`](#),
    [`StopOnNaNLoss`](#), and [`MetricsPrinter`](#).
- `cbrunner = LinearRunner()`: Callback runner to use.

## Fields

*(Use this as a reference when implementing callbacks)*

- `model`, `optimizer`, and `lossfn` are stored as passed in
- `data` is a `PropDict` of data iterators, usually `:training` and `:validation`.
- `params`: An instance of `model`'s parameters of type `Flux.Params`. If `model` is
    a `NamedTuple`, then `params` is a `NamedTuple` as well.
- `step::`[`PropDict`](#): State of the last step. Contents depend on the last run
    [`Phase`](#).
- `cbstate::`[`PropDict`](#): Special state container that callbacks can
    save state to for other callbacks. Its keys depend on what callbacks
    are being used. See the [custom callbacks guide](/documents/docs/callbacks/custom.md)
    for more info.
- `metadata::`[`PropDict`](#): A container to hold useful metadata associated
    with the `Learner`. Defaults to an empty container for standard training.
    This can be useful for storing hyper-parameters for
    unconventional custom training methods.
"""
function Learner(model, lossfn; callbacks = [], data = (), optimizer = ADAM(), kwargs...)
    return Learner(model, data, optimizer, lossfn, callbacks...; kwargs...)
end


"""
    Learner(model, data, optimizer, lossfn, [callbacks...; kwargs...])
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
        setupoptimstate(model, optimizer),
        PropDict(),
        cbs,
        PropDict(),
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
]


#  Callback handling
handle(event, learner, phase) = handle(learner.callbacks.runner, event, phase, learner)


# Other

phasedataiter(::AbstractTrainingPhase) = :training
phasedataiter(::AbstractValidationPhase) = :validation


function model!(learner, model)
    learner.model = model
    learner.params = setupoptimstate(model, learner.optimizer)
end

# Flux.jl optimisers store `params`, while Optimisers.jl store the result of `setup`
setupoptimstate(model, ::Flux.Optimise.AbstractOptimiser) = Flux.params(model)
# Optimisers.jl has no abstract supertype so we assume non-Flux optimisers
# conform to the Optimisers.jl interface.
setupoptimstate(model, optim) = Optimisers.setup(optim, model)


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
