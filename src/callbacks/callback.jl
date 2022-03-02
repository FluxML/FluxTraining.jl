
"""
    abstract type AbstractCallback

Supertype of [`SafeCallback`](#)/`Callback`. When implementing callbacks,
you should subtype [`SafeCallback`](#) instead.
"""
abstract type AbstractCallback end

abstract type SafeCallback <: AbstractCallback end

"""
    abstract type Callback

Supertype of all callbacks. Callbacks add custom functionality to
the training loop by hooking into different [`Events.Event`](#)s

Any `Callback` can be used by passing it to [`Learner`](#). See
`subtypes(FluxTraining.Callback)` for implementations.

## Extending

See [Custom callbacks](/documents/docs/callbacks/custom.md) for a less
succinct tutorial format.

1. Create a `struct MyCallback` that subtypes `FluxTraining.Callback`.
2. Add event handlers by implementing methods for
   [`on`](#)`(event, phase, callback, learner)`. Methods should always
   dispatch on your callback, and may dispatch on specific [`Phases.Phase`](#)s
   and [`Events.Event`](#)s.

   For example, to implement an event handler that runs at the end of every
   step during training: `on(::StepEnd, ::AbstractTrainingPhase, ::MyCallback, learner)`.
3. Define what state the callback accesses and/or modifies by implementing
   [`stateaccess`](#)`(::MyCallback)`. While `learner` is always passed as an
   argument to `on` event handlers, by default a callback can not read or write
   to its fields. See [`stateaccess`](#) for more detail.

   If a callback needs to write some state that other callbacks should be able
   to access, it can store it in `learner.cbstate` if you add a permission in
   `stateaccess`.
4. If the callback needs some one-time initialization, you can implement [`init!`](#)
   which will be run at least once before any step is run.
"""
const Callback = SafeCallback

abstract type UnsafeCallback <: AbstractCallback end


# TODO: implement proper merging of permissions

"""
    stateaccess(callback)

Return a named tuple determining what learner state `callback`
can access. The default is `(;)`, the empty named tuple, meaning
no state can be accessed. Implementations of `stateaccess` should
always return the least permissions possible.

## Extending

For example, the [`ToGPU`](#) callback needs to write both the model and
the batch data, so its `stateaccess` implementation is:

```julia
stateaccess(::ToGPU) = (
    model = Write(),
    params = Write(),
    step = (xs = Write(), ys = Write()),
)
```

When defining `stateaccess`, be careful that you do return a `NamedTuple`.
`(x = Read(),)` is one but `(x = Read())` (without the comma) is parsed as
an assignment with value `Read()`.
"""
stateaccess(::Callback) = (;)
runafter(::AbstractCallback) = ()

"""
    abstract type ConflictResolution

A conflict resolution strategy for resolving write/write conflicts
of two callbacks.

See [`resolveconflict`](#).
"""
abstract type ConflictResolution end

"""
    abstract type NotDefined <: ConflictResolution

The default implementation of [`resolveconflict`](#). If a conflict
is detected, this ensures an error message is printed.
"""
struct NotDefined <: ConflictResolution end

"""
    abstract type Unresolvable <: ConflictResolution

Return from [`resolveconflict`](#) to indicate that two callbacks
are incompatible and cannot be used together.
"""
struct Unresolvable <: ConflictResolution end

"""
    abstract type RunFirst <: ConflictResolution

Return `RunFirst(cb1/cb2)` from [`resolveconflict`](#)`(cb1, cb2)` to indicate
that one of the callbacks should always run before the other.
"""
struct RunFirst <: ConflictResolution cb end

"""
    abstract type NoConflict <: ConflictResolution

Return from [`resolveconflict`](#) to indicate that, while the callbacks
modify the same state, they can be used together without any problems.
"""
struct NoConflict <: ConflictResolution end

function _resolveconflict(cb1, cb2)
    r = resolveconflict(cb1, cb2)
    if r === NotDefined()
        return resolveconflict(cb2, cb1)
    else
        return r
    end
end

"""
    resolveconflict(cb1, cb2)

Define a conflict resolution strategy for resolving a write/write conflict
between two callbacks.

The default is [`NotDefined()`], which will result in an error and a message
to implement this method.

To implement, dispatch on the callback types that you which to resolve (in any
order) and return one of the following:

- [`Unresolvable`](#)`()` if the callbacks must not be used together
- [`RunFirst`](#)`(cb)` if one of the callbacks needs to run first; or
- [`NoConflict`](#)`()` if the callbacks may run together in any order
"""
resolveconflict(::AbstractCallback, ::AbstractCallback) = NotDefined()


"""
    abstract type FitException

Abstract types for exceptions that can be thrown during fitting,
to change its control flow.

See [`CancelStepException`](#), [`CancelEpochException`](#), [`CancelFittingException`](#).
"""
abstract type FitException <: Exception end


"""
    CancelStepException(message)

Throw during fitting to cancel the currently running step.
This prematurely ends the current step without throwing an error.
Must be thrown inside the context of [`runstep`](#).

## Examples

```julia
runepoch(learner, phase) do _
    for (xs, ys) in batches
        runstep(learner, phase, (; xs, ys)) do _, state
            # training logic...
            if isnan(state.loss).
                throw(CancelStepException("Skipping NaN loss"))
            end
        end
    end
end
```
"""
struct CancelStepException <: FitException
    msg::String
end


"""
    CancelEpochException(message)

Throw during fitting to cancel the currently running epoch.
This prematurely ends the current epoch without throwing an error.
Must be thrown inside the context of [`runepoch`](#).

## Examples

```julia
runepoch(learner, phase) do _
    for batch in batches
        step!(learner, phase, batch)
        if learner.step.loss < 1.
            throw(CancelEpochException("Reached target loss"))
        end
    end
end
```
"""
struct CancelEpochException <: FitException
    msg::String
end


"""
    CancelFittingException(msg)

Throw during fitting to cancel it.
"""
struct CancelFittingException <: FitException
    msg::String
end


# Callback hook

"""
    on(event::Event, phase::Phase, callback::AbstractCallback, learner)

Handle `event` with [`Callback`](#) `callback`.
By default, this event handler does nothing for a callback.

To see events which an `AbstractCallback` handles, use

```julia
methods(Training.on, (Any, Any, MyCallbackType, Any)
```

## Extending

You can add event handlers to [`Callback`](#)s by implementing a method for `on`.
See also [`Callback`](#) and [custom callbacks](documents/docs/callbacks/custom.md).

A method of `on` should *always* dispatch on the callback type, i.e.
`on(event, phase, cb::MyCallback, learner)`. It may also dispatch on specific
[`Event`](#)s and [`Phase`](#). It should not dispatch on a specific type for
`learner`.
"""
on(::Event, phase, ::Callback, learner) = return

_on(e, p, cb, learner) = on(e, p, cb, learner)
function _on(e, p, cb::SafeCallback, learner)
    perms = Zygote.ignore() do
        stateaccess(cb)
    end
    on(e, p, cb, protect(learner, perms))
end


"""
    init!(callback, learner)

Initialize a callback. Default is to do nothing.

## Extending

To extend for a callback, implement `init!(cb::MyCallback, learner)`.
`init!` can set up internal state of a callback that depends on `learner`
and can also initialize shared callback state in `learner.cbstate`.
Just like `on` event handlers, the state access permissions must be correctly
defined using [`stateaccess`](#) to do so.

`init!` must also be idempotent, i.e. running it twice on the same `Learner`
should have the same effect as runnning it once.
"""
init!(::Callback, learner) = return
