
abstract type AbstractCallback end
abstract type SafeCallback <: AbstractCallback end
"""
    abstract type Callback

Callbacks can add custom functionality to the training loop.
See [custom callbacks](../docs/callbacks/custom.md) for more info.
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

For example, the [`ToGPU`](#) callback needs to write both the model and
the batch data, so its `stateaccess` implementation is:

```julia
stateaccess(::ToGPU) = (
    model = Write(),
    params = Write(),
    batch = (xs = Write(), ys = Write()),
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
struct NotDefined <: ConflictResolution end
struct Unresolvable <: ConflictResolution end
struct RunFirst <: ConflictResolution cb end
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

See [`CancelBatchException`](#), [`CancelEpochException`](#), [`CancelFittingException`](#).
"""
abstract type FitException <: Exception end
"""
    CancelBatchException(msg)

Throw during fitting to cancel the currently running batch.
"""
struct CancelBatchException <: FitException
    msg::String
end


"""
    CancelEpochException(msg)

Throw during fitting to cancel the currently running epoch.
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

Handle `event` with `callback`. Can dispatch on an `Phase` and
receives `learner` as an additional argument.

If not overwritten with a more specific method, does nothing.

To see events which an `AbstractCallback` handles, use

```julia
methods(Training.on, (Any, Any, MyCallbackType, Any)
```
"""
on(::Event, phase, ::Callback, learner) = return

_on(e, p, cb, learner) = on(e, p, cb, learner)
function _on(e, p, cb::SafeCallback, learner)
    perms = Zygote.ignore() do
        stateaccess(cb)
    end
    on(e, p, cb, protect(learner, perms))
end
