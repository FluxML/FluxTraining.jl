
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
can access.

The default is `(;)`, the empty named tuple, meaning no state can be accessed.

Implementations of `stateaccess` should always return the least permissions
possible.

For example, the [`ToGPU`](#) callback needs to write both the model and the batch data,
so its `stateaccess` implementation is:

```julia
stateaccess(::ToGPU) = (
    model = Write(),
    params = Write(),
    batch = (xs = Write(), ys = Write()),
)
```

Be careful when defining `stateaccess` that you do return a `NamedTuple`. `(x = Read(),)` is
one but `(x = Read())` (without the comma) is parsed as an assignment with value `Read()`.
"""
stateaccess(::Callback) = (;)
runafter(::AbstractCallback) = ()

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
resolveconflict(::AbstractCallback, ::AbstractCallback) = NotDefined()


abstract type FitException <: Exception end
struct CancelBatchException <: FitException
    msg::String
end
struct CancelEpochException <: FitException
    msg::String
end
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
