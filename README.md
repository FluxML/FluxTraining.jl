# FluxTraining

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lorenzoh.github.io/FluxTraining.jl/dev)

A powerful, extensible neural net training library inspired by fast.ai

`FluxTraining` gives you an endlessly extensible training loop for deep learning. It is inspired by [fast.ai](https://docs.fast.ai)

It exposes a small set of [extensible](docs/extending.md) interfaces and uses them to implement

- hyperparameter scheduling
- metrics
- logging
- training history; and
- model checkpointing

## Getting started

The best place to start is the [user guide](docs/guide/getting_started.md).

See also the [reference](docstrings.md).

{#ecosystem}
## Ecosystem

Unlike fastai, `FluxTraining` focuses on the training part of the deep learning pipeline. Other packages you may find useful are

- [`Metalhead.jl`](https://github.com/FluxML/Metalhead.jl) and [`FluxModels.jl`](https://github.com/lorenzoh/FluxModels.jl) for models
- [`Augmentor.jl`](https://github.com/Evizero/Augmentor.jl) and [`DataAugmentation.jl`](https://github.com/lorenzoh/DataAugmentation.jl) for data augmentation
- [`DataLoaders.jl`](https://github.com/lorenzoh/DataLoaders.jl) for parallel data loading; and
- [`DLDatasets.jl`](https://github.com/lorenzoh/DLDatasets.jl) for datasets
