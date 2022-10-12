# Models

FluxTraining.jl works with all [Flux.jl](https://github.com/FluxML/Flux.jl)-compatible models. Unless you are using a [custom training loop](/doc/docs/tutorials/training.md), a `model` is expected to take a single input `xs`, which corresponds to the encoded inputs returned by your [data iterator](/doc/docs/background/dataiterator.md). This means the following has to work:

```julia
xs, ys = first(dataiter)
ŷs = model(xs)
```

`model` also has to be differentiable. If you're composing Flux.jl layers, this is likely the case. You can always make sure by testing:


```julia
using Flux, Zygote

xs, ys = first(dataiter)
lossfn = Flux.mse
grads = Zygote.gradient(Flux.params(model)) do
    lossfn(model(xs), ys)
end
```

## Creating models

The simplest way to create a Flux.jl-compatible model is to use layers from Flux.jl. 
A good entrypoint is [this tutorial](https://fluxml.ai/Flux.jl/stable/models/basics/)in Flux's documentation.

There is also a large number of packages that provide complete model architectures or domain-specific layers. Below is a non-exhaustive list:

- [Metalhead.jl](https://github.com/FluxML/Metalhead.jl) implements common model architectures for computer vision,
- [GraphNeuralNetworks.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) provides layers and utilities for graph neural networks,
- [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) implements transformer models including pretrained language models

