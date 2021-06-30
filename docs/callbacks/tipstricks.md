{cell=main, output=false, result=false, style="display:none"}
```julia
using FluxTraining
using FluxTraining.Events, FluxTraining.Phases
```

# Tips & tricks



## Listing event handlers for a callback

Use `Base.methods` to check what events a callback handles:

{cell=main}
```julia
methods(FluxTraining.on, (Any, Any, Recorder, Any))
```


## Visualize the callback dependency graph

You can use [*GraphPlot.jl*](https://juliagraphs.github.io/GraphPlot.jl/) to visualize the dependencies between callbacks:

{cell=main, output=false, result=false, style="display:none"}
```julia
learner = Learner(
    nothing, (nothing, nothing), nothing, nothing,  # dummy arguments
    ToGPU(),
);
```

{cell=main}
```julia
using GraphPlot
gplot(learner.callbacks.graph, nodelabel = learner.callbacks.cbs, layout = stressmajorize_layout)
```

*(the target of an arrow depends on the origin)*

As an example for a detected dependency, we can see that [`MetricsPrinter`](#) runs after [`Metrics`](#). [`MetricsPrinter`](#) prints the values of all metrics, so [`Metrics`] needs to run first.