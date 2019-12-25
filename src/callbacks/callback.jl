import Base.Order: ReverseOrdering


mutable struct TrainingState
    learner::Learner
    data
    opt
    nepochs
    epoch
    iteration
    batchx
    batchy
    lossbatch
    output
    exception
    gradients
end

TrainingState() = TrainingState(
    nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing,
    nothing, nothing)


# Callbacks

abstract type AbstractCallback end

priority(c::AbstractCallback) = 0

on_train_begin(c::AbstractCallback, state::TrainingState) = return
on_train_end(c::AbstractCallback, state::TrainingState) = return

on_epoch_begin(c::AbstractCallback, state::TrainingState) = return
on_epoch_end(c::AbstractCallback, state::TrainingState) = return

on_batch_begin(c::AbstractCallback, state::TrainingState) = return
on_batch_end(c::AbstractCallback, state::TrainingState) = return

on_loss_begin(c::AbstractCallback, state::TrainingState) = return

on_backward_begin(c::AbstractCallback, state::TrainingState) = return
on_backward_end(c::AbstractCallback, state::TrainingState) = return

# Callback handling

struct CallbackHandler
    callbacks
    state::TrainingState
    CallbackHandler(callbacks) = new(
        sort!(callbacks, by=priority),
        TrainingState()
    )
end

function on_train_begin(ch::CallbackHandler, nepochs::Integer, opt, data)
    ch.state.nepochs = nepochs
    ch.state.opt = opt
    ch.state.data = data
    ch.state.epoch = 0
    ch.state.iteration = 0
    ch.state.batch = 0
    foreach(cb -> on_train_begin(cb, ch.state), ch.callbacks)
end

function on_train_end(ch::CallbackHandler, exception::Union{Nothing, Exception})
    ch.state.exception = exception
    foreach(cb -> on_train_end(cb, ch.state), ch.callbacks)
end

function on_epoch_begin(ch::CallbackHandler)
    ch.state.epoch += 1
    foreach(cb -> on_epoch_begin(cb, ch.state), ch.callbacks)
end

function on_epoch_end(ch::CallbackHandler)
    foreach(cb -> on_epoch_end(cb, ch.state), ch.callbacks)
end

function on_batch_begin(ch::CallbackHandler, batch)
    ch.state.batch = batch
    ch.state.iteration += 1
    foreach(cb -> on_batch_begin(cb, ch.state), ch.callbacks)
    return ch.state.batch
end

function on_batch_end(ch::CallbackHandler)
    foreach(cb -> on_batch_end(cb, ch.state), ch.callbacks)
end

function on_loss_begin(ch::CallbackHandler, output)
    ch.state.output = output
    foreach(cb -> on_loss_begin(cb, ch.state), ch.callbacks)
end

function on_backward_begin(ch::CallbackHandler, lossbatch)
    ch.state.lossbatch = lossbatch
    foreach(cb -> on_backward_begin(cb, ch.state), ch.callbacks)
end

function on_backward_end(ch::CallbackHandler, gradients)
    ch.state.gradients = gradients
    foreach(cb -> on_backward_end(cb, ch.state), ch.callbacks)
end
