{cell=main, output=false, result=false, style="display:none"}
```julia
using FluxTraining
using FluxTraining: Callback, Read, Write, stateaccess
model, data, lossfn, optimizer = nothing, nothing, nothing, nothing
```

# How to use callbacks

Callbacks allow injecting functionality at many points during the training loop.

To use them, simply pass each callback to `Learner`:

{cell=main, output=false}
```julia
learner = Learner(
    model, data, optimizer, lossfn,  # required arguments
    ToGPU(), Metrics(accuracy))      # pass any number of callbacks as additional arguments
```

Some useful callbacks are added by default:

{cell=main, output=false}
```julia
learner.callbacks.cbs
```

See [callback reference](./reference.md) for a list of all callbacks included in *FluxTraining.jl* and their documentation.
    
!!! info "Ordering"

    The order the callbacks are passed in doesn't matter. *FluxTraining.jl* creates a dependency graph that makes sure the callbacks are run in the correct order. Read [custom callbacks](./custom.md) to find out how to create callbacks yourself.