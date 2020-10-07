

mutable struct Callbacks
    cbs::Vector
    runner::CallbackRunner
    graph::SimpleDiGraph
    initialized::Bool
end
Callbacks(cbs, runner = LinearRunner()) = Callbacks(cbs, runner, callbackgraph(cbs), false)


"""
    $TYPEDEF

Stores data of the last processed batch.

$FIELDS

(!) If used in callbacks, some fields may be `nothing` as
they are reset after every step.
"""
# TODO: maybe make this dependent on the current `Phase`?
@with_kw mutable struct BatchState
    xs = nothing
    ys = nothing
    ŷs = nothing
    loss = nothing
    grads = nothing
end


mutable struct Learner
    model
    data
    optimizer
    lossfn
    params
    batch::BatchState
    callbacks::Callbacks
    cbstate::Dict
end


"""
    Learner(model, data, optimizer, lossfn, [callbacks...; kwargs...])

Holds and coordinates all state of the training. `model` is trained by
optimizing `lossfn` with `optimizer` on `data`.

## Arguments

- `model`
- `data`: Tuple of data iterators in the order `(traindata, valdata, [testdata])`.
    Must be iterable and return tuples of `(xs, ys)`
- `lossfn`: Function with signature `lossfn(model(x), y) -> Number`
- `optimizer`
- `callbacks...`: Any other unnamed arguments are callbacks

## Keyword arguments

- `usedefaultcallbacks = true`: Whether to add some basic callbacks. Included
    are [`Loss`](#), [`Recorder`](#), [`ProgressBarLogger`](#),
    [`StopOnNaNLoss`](#), and [`MetricsLogger`](#)
- `cbrunner = LinearRunner()`: Callback runner to use.

## Fields

- `model`, `optimizer`, and `lossfn` are stored as passed in
- `data` is a `NamedTuple` of `(training = ..., validation = ..., test = ...)`.
    Some values might be `nothing` if you didn't pass in multiple data iterators.
- `params`: an instance of `model`'s parameters of type `Flux.Params`
- `batch`: State of the current batch, including:

    - `batch.xs`: model inputs
    - `batch.ys`: target outputs
    - `batch.ŷs`: model outputs, i.e. `model(xs)`
    - `batch.loss`: batch loss, i.e. `lossfn(ŷs, ys)`
    - `batch.gs`: batch gradients, instance of `Zygote.Grads`

    (!) Note: Depending on the progress of the step, some fields may be `nothing`,
    e.g. the `gs` before the backward pass.
- `cbstate::Dict{Symbol,Any}`: Special state container that callbacks can
    save state to for other callbacks. Its keys depend on what callbacks
    are being used. See the [custom callbacks guide](../docs/callbacks/custom.md)
    for more info.


"""
function Learner(
        model, data, optimizer, lossfn, callbacks...;
        usedefaultcallbacks = true, cbrunner = LinearRunner()
    )
    callbacks = collect(Callback, callbacks)
    if usedefaultcallbacks
        for cb in defaultcallbacks()
            if !any(typeof(cb) .== typeof.(callbacks))
                push!(callbacks, cb)
            end
        end
    end

    return Learner(
        model, dataiters(data), optimizer, lossfn,
        Flux.params(model),
        BatchState(),
        Callbacks(callbacks, cbrunner),
        Dict{Symbol, Any}())
end

Base.show(io::IO, learner::Learner) = print(io, "Learner()")

defaultcallbacks()::Vector{AbstractCallback} = [
    ProgressBarLogger(),
    MetricsLogger(),
    StopOnNaNLoss(),
    Recorder(),
    Loss(),
]


#  Callback handling
handle(event, learner, phase) = handle(learner.callbacks.runner, event, phase, learner)


# Other

getdataiter(phase::AbstractTrainingPhase, learner) = learner.data.training
getdataiter(phase::ValidationPhase, learner) = learner.data.validation


function model!(learner, model)
    learner.model = model
    learner.params = Flux.params(model)
end


numsteps(learner::Protected, phase) = numsteps(getfield(learner, :data), phase)
numsteps(learner, phase) = length(getdataloader(phase, learner))

hascallback(learner::Protected, T) = hascallback(getfield(learner, :data), phase)
hascallback(learner, T) = any(C <: T for C in typeof.(learner.callbacks.cbs))


dataiters(train, val = nothing, test = nothing) = (training = train, validation = val, test = test)
dataiters(t::Tuple) = dataiters(t...)
dataiters(t::NamedTuple) = keys(t) == (:training, :validation, :test) ? t : error("Wrong keys.")
# TODO: fix

#=
function setschedule!(learner, schedule)
    learner.scheduler = ParamScheduler(
        delayschedule(schedule, learner.recorder.epoch)
    )
end
=#


# TODO: move to Callback

#=
function artifactpath(learner)
    p = joinpath(pwd(), ".artifacts", string(learner.runid))
    if !isdir(p)
        mkpath(p)
    end
    return p
end
=#
