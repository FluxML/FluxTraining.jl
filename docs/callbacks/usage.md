{cell=main, output=false, result=false, style="display:none"}
```julia
using FluxTraining
using FluxTraining: Callback, Read, Write, stateaccess
```

# Using callbacks

Callbacks allow injecting functionality at many points during the training loop.

To use them, pass a list of callbacks to `Learner`:

{cell=main, output=false}
```julia
learner = Learner(
    nothing, (nothing, nothing), nothing, nothing;  # dummy arguments
    callbacks = [
        ToGPU(),
    ]
);
```

Some useful callbacks are added by default:

{cell=main, output=false}
```julia
learner.callbacks.cbs
```

See [callback reference](./reference.md) for a list of all callbacks included in *FluxTraining.jl* and their documentation.
    
!!! info "Ordering"

    The order the callbacks are passed in doesn't matter. *FluxTraining.jl* creates a dependency graph that makes sure the callbacks are run in the correct order. Read [custom callbacks](./custom.md) to find out how to create callbacks yourself.