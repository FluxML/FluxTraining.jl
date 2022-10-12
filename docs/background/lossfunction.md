# Loss functions

A loss function compares model outputs to true targets, resulting in a loss. For a loss function to be compatible with the standard supervised training loop, the following properties must hold.

Firstly, the loss function should accept the model outputs and targets, and return a single scalar value. Given a [data iterator](/doc/docs/background/dataiterator.md) `dataiter` and a [model](/doc/docs/background/model.md) `model`:

```julia
xs, ys = dataiter
ŷs = model(xs)
lossfn(ŷs, ys) isa Number
```

The loss function must also be differentiable, so that gradients can be calculated during training. See [models](/doc/docs/background/model.md) for more on how to check this.

## Creating loss functions

Flux.jl comes with a lot of commonly used loss functions built-in in its submodule `Flux.Losses`. See [Flux.jl loss functions](https://fluxml.ai/Flux.jl/stable/models/losses/) for a complete reference.

You can also write your own loss functions. If you are using non-mutating array operations, there is a good chance that it will be differentiable and also be compatible with GPU arrays from [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).