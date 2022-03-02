# FluxTraining.jl

[Docs (master)](https://fluxml.ai/FluxTraining.jl/dev/i)

A Julia package for using and writing powerful, extensible training loops for deep learning models.

## What does it do?

- Implements common training loops to take the boilerplate out of training deep learning models
- Lets you add features to training loops through reusable callbacks
- Comes with callbacks for many common use cases like hyperparameter scheduling, metrics tracking, checkpointing, early stopping, and more
- Is extensible by creating custom, reusable callbacks or even custom training loops

## When should you use FluxTraining.jl?

- You don't want to implement your own metrics tracking and hyperparameter scheduling or _insert common training feature here_ for the 10th time
- You want to use composable and reusable components that enhance your training loop
- You want a simple training loop with reasonable defaults that can grow to the needs of your project


### How do you use it?

Install like any other Julia package using the package manager:

```julia-repl
]add FluxTraining
```

After installation, import it, create a `Learner` from a [Flux.jl](https://github.com/FluxML/Flux.jl) model, data iterators, an optimizer, and a loss function. Finally train with [`fit!`](#).

```julia
using FluxTraining

learner = Learner(model, (trainiter, validiter), optim, lossfn)
fit!(learner, 10)
```

## Next, you may want to read

- [Getting started](docs/getting_started.md)
- [A full example training an image classifier on the MNIST dataset](docs/tutorials/mnist.ipynb)
- The [FastAI.jl documentation](https://fluxml.github.io/FastAI.jl/dev) which features many end-to-end examples

## Acknowledgements

The design of FluxTraining.jl's two-way callbacks is adapted from [fastai](https://docs.fast.ai)'s training loop.
