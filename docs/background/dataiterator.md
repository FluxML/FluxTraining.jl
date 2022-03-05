# Data iterators

A data iterator is an iterator over batches of the data that is used for one step of fitting. 
You can use different data iterators with this package, as long as they have the following properties.

Firstly, you must be able to iterate over a data iterator:

```julia
for batch in dataiter
    # step
end
```

The data iterator must also be compatible with the other components of the [`Learner`](#). For the standard supervised learning step ([`TrainingPhase`](#) and [`ValidationPhase`](#)), this means

- `batch` is a tuple `(xs, ys)` of encoded inputs and targets,
- `xs` is a valid input to the [model](/documents/docs/background/model.md), so `ŷs = model(xs)`; and
- `ys` can be compared to the model output with the [loss function](/documents/docs/background/lossfunction.md), i.e. `lossfn(ŷs, ys)`

If you are working with a [custom training loop](/documents/docs/tutorials/training.md), you may need to satisfy additional or different properties.

## Creating data iterators

The simplest data iterator is a vector of preloaded batches. This is what we're using in the [MNIST tutorial](/documents/docs/tutorials/mnist.ipynb). This is a fine approach for smaller datasets, but has some limitations.

First of all, there is no principled way for doing things like splitting, subsetting and shuffling data. For this, we recommend using [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl) which provides this functionality and many more utilities for defining and working with datasets.

Another issue is that of memory load: if the whole dataset is too large to be preloaded in to memory, we have to load individual batches during training. To do this in a way that doesn't slow down the training itself, we suggest using [DataLoaders.jl](https://github.com/lorenzoh/DataLoaders.jl). DataLoaders.jl is compatible with MLDataPattern.jl and allows you to easily create efficient data iterators for out-of-memory datasets. The documentation of DataLoaders.jl also has a lot more information on working with large dataset for deep learning.

