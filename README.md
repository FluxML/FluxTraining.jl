# FluxTraining.jl

[Docs (master)](https://lorenzoh.github.io/FluxTraining.jl/dev)

A powerful, extensible neural net training library.


*FluxTraining.jl* gives you an endlessly extensible training loop for deep learning. It is inspired by [fastai](https://docs.fast.ai).

It exposes a small set of extensible interfaces and uses them to implement

- hyperparameter scheduling
- metrics
- logging
- training history; and
- model checkpointing

Install using `]add FluxTraining`.

Read [getting started](docs/getting_started.md) first and the [user guide](docs/overview.md) if you want to know more. See also the [reference](docstrings.md) for detailed function documentation.


*FluxTraining.jl* is part of an ongoing effort to improve Julia's deep learning infrastructure and will be the training library for the work-in-progress [*FastAI.jl*](https://github.com/FluxML/FastAI.jl). Drop by on the [Julia Zulip](julialang.zulipchat.com) and say hello in the stream `#ml-ecosystem-coordination`.