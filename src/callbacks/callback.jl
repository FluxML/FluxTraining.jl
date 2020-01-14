
mutable struct TrainingState
    learner
    params
    nepochs
    epoch
    step
    batchx
    batchy
    lossbatch
    output
    exception
    gradients
end

TrainingState() = TrainingState(
    nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing,)


# Callbacks

# TODO: make Callbacks parametric to FittingPhases
abstract type AbstractCallback end

order(c::Type{<:AbstractCallback}) = 0

on_train_begin(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return
on_train_end(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return

on_epoch_begin(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return
on_epoch_end(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return

on_batch_begin(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return
on_batch_end(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return

on_loss_begin(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return

on_backward_begin(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return
on_backward_end(c::AbstractCallback, state::TrainingState, phase::AbstractFittingPhase ) = return

# Callback handling

mutable struct CallbackHandler
    callbacks
    state::TrainingState
    phase::AbstractFittingPhase
end

CallbackHandler(callbacks, state = TrainingState()) = CallbackHandler(
    sort!(callbacks, by=cb -> order(typeof(cb))),
    state,
    UninitializedPhase()
)

function on_train_begin(ch::CallbackHandler, nepochs::Integer, learner)
    ch.state.nepochs = nepochs
    ch.state.learner = learner
    foreach(cb -> on_train_begin(cb, ch.state, ch.phase), ch.callbacks)
end

function on_train_end(ch::CallbackHandler, exception::Union{Nothing, Exception})
    ch.state.exception = exception
    foreach(cb -> on_train_end(cb, ch.state, ch.phase), ch.callbacks)
end

function on_epoch_begin(ch::CallbackHandler, phase::AbstractFittingPhase)
    ch.phase = phase
    if phase isa TrainingPhase
        ch.state.epoch += 1
        ch.state.step = 0
    end
    foreach(cb -> on_epoch_begin(cb, ch.state, ch.phase), ch.callbacks)
end

function on_epoch_end(ch::CallbackHandler)
    foreach(cb -> on_epoch_end(cb, ch.state, ch.phase), ch.callbacks)
end

function on_batch_begin(ch::CallbackHandler, batch)
    ch.state.batchx, ch.state.batchy = batch |> cpu
    if ch.phase isa TrainingPhase
        ch.state.step += 1
    end
    foreach(cb -> on_batch_begin(cb, ch.state, ch.phase), ch.callbacks)
    return ch.state.batchx, ch.state.batchy
end

function on_batch_end(ch::CallbackHandler)
    foreach(cb -> on_batch_end(cb, ch.state, ch.phase), ch.callbacks)
end

function on_loss_begin(ch::CallbackHandler, output)
    ch.state.output = output
    foreach(cb -> on_loss_begin(cb, ch.state, ch.phase), ch.callbacks)
end

function on_backward_begin(ch::CallbackHandler, lossbatch)
    ch.state.lossbatch = lossbatch
    foreach(cb -> on_backward_begin(cb, ch.state, ch.phase), ch.callbacks)
end

function on_backward_end(ch::CallbackHandler, gradients)
    ch.state.gradients = gradients
    foreach(cb -> on_backward_end(cb, ch.state, ch.phase), ch.callbacks)
end
