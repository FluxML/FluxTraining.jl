# Training loop

FluxTraining.jl comes with a training loop for standard supervised learning problems, but for different tasks like self-supervised learning, being able to write custom training logic is essential. The package's training loop API requires little boilerplate to convert a regular Flux.jl training loop while making it compatible with existing callbacks.

## Supervised training, step-by-step

We'll explore the API step-by-step by converting a basic training loop and then discuss ways in which more complex training loops can be implemented using the same approach. The central piece of a training loop is the logic for a single training step, and in many cases, that will be all you need to implement. Below is the definition of a basic vanilla Flux.jl training step. It takes a batch of data, calculates the loss, gradients and finally updates the parameters of the model.

```julia
function step!(model, batch, params, optimizer, lossfn)
    xs, ys = batch
    grads = gradient(params) do
        ŷs = model(xs)
        loss = lossfn(ŷs, ys)
        return loss
    end
    update!(optimizer, params, grads)
end
```

To make a training step work with FluxTraining.jl and its callbacks, we need to

- store data for a step so that callbacks can access it (e.g. [`Metrics`](#) uses `ys` and `ŷs` to evaluate metrics for each step); and
- dispatch events so the callbacks are triggered

We first need to create a [`Phase`](#) and implement a method for [`FluxTraining.step!`](#) that dispatches on the phase type. `Phase`s are used to define different training behaviors using the same API and to define callback functionality that is only run during certain phases. For example, [`Scheduler`](#) only runs during [`AbstractTrainingPhase`](#)s but not during [`ValidationPhase`](#). Let's implement such a phase and method, moving the arguments inside a [`Learner`](#) in the process.

```julia
struct MyTrainingPhase <: FluxTraining.AbstractTrainingPhase

function FluxTraining.step!(learner, phase::MyTrainingPhase, batch)
    xs, ys = batch
    grads = gradient(learner.params) do
        ŷs = learner.model(xs)
        loss = learner.lossfn(ŷs, ys)
        return loss
    end
    update!(learner.optimizer, learner.params, grads)
end
```

Now we can already train a model using this implementation, for example using [`epoch!`](#)`(learner, MyTrainingPhase(), dataiter)`. However, no callbacks would be called, since we haven't yet put in any logic that dispatches events or stores the step state. We can do both by using the helper function [`runstep`](#) which takes care of runnning our step logic, dispatching a [`StepBegin`](#) and [`StepEnd`](#) event before and after and handling control flow exceptions like [`CancelStepException`](#). Additionally, `runstep` gives us a function `handle` which we can use to dispatch events inside the step, and `state` a container for storing step state. Let's use `runstep` and store the variables of interest inside `state`:

```julia
function step!(learner, phase::MyTrainingPhase, batch)
    step(learner, phase, batch) do handle, state
        state.xs, state.ys = batch
        state.grads = gradient(learner.params) do
            state.ŷs = learner.model(state.xs)
            state.loss = learner.lossfn(state.ŷs, state.ys)
            return loss
        end
        update!(learner.optimizer, learner.params, grads)
    end
end
```

Now callbacks like [`Metrics`](#) can access variables like `ys` through `learner.step` (which is set to the last `state`). Finally, we can use `handle` to dispatch additional events:

```julia
using FluxTraining.Events: LossBegin, BackwardBegin, BackwardEnd

function step!(learner, phase::MyTrainingPhase, batch)
    step(learner, phase, batch) do handle, state
        state.xs, state.ys = batch
        state.grads = gradient(learner.params) do
            state.ŷs = learner.model(state.xs)
            handle(LossBegin())
            state.loss = learner.lossfn(state.ŷs, state.ys)
            handle(BackwardBegin())
            return loss
        end
        handle(BackwardEnd())
        update!(learner.optimizer, learner.params, grads)
    end
end
```

The result is the full implementation of FluxTraining.jl's own [`TrainingPhase`](#)! Now we can use `epoch!` to train a `Learner` with full support for [all callbacks](../callbacks/reference.md):

```julia
for i in 1:10
    epoch!(learner, MyTrainingPhase(), dataiter)
end
```

## Validation

The implementation of [`ValidationPhase`](#) is even simpler; it runs the forward pass and stores variables so that callbacks like [`Metrics`](#) can access them.

```julia
struct ValidationPhase <: AbstractValidationPhase end

function step!(learner, phase::ValidationPhase, batch)
    runstep(learner, phase) do _, state
        state.xs, state.ys = batch
        state.ŷs = learner.model(state.xs)
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end
```

## Epoch logic

We didn't need to implement a custom `epoch!` method for our phase since the default is fine here: it just iterates over every batch and calls `step!`. In fact, let's have a look at how `epoch!` is implemented:

```julia
function epoch!(learner, phase::Phase, dataiter)
    runepoch(learner, phase) do handle
        for batch in dataiter
            step!(learner, phase, batch)
        end
    end
end
```

Here, [`runepoch`](#), similarly to `runstep`, takes care of epoch start/stop events and control flow. If you want more control over your training loop, you can use it to write training loops that directly use `step!`:

```julia
phase = MyTrainingPhase()
withepoch(learner, phase) do handle
    for batch in dataiter
        step!(learner, phase, batch)
        if learner.step.loss < 0.1
            throw(CancelFittingException("Low loss reached."))
        end
    end
end
```


## Tips

Here are some additional tips for making it easier to implement complicated training loops.

- You can pass (named) tuples of models to the `Learner` constructor. For example, for generative adversarial training, you can pass in `(generator = ..., critic = ...)` and then refer to them inside the `step!` implementation, e.g. using `learner.model.generator`. The models' parameters will have the same structure, i.e. `learner.params.generator` corresponds to `params(learner.model.generator)`.
- You can store any data you want in `state`.
- When defining a custom phase, instead of subtyping `Phase` you can subtype [`AbstractTrainingPhase`](#) or [`AbstractValidationPhase`](#) so that some context-specific callbacks will work out of the box with your phase type. For example, [`Scheduler`](#) sets hyperparameter values only during `AbstractTrainingPhase`.