# Custom callbacks

*FluxTraining.jl*'s callback system is built around multiple dispatch, so you specify which part of the training you want to "hook into" by dispatching on [`Phase`](#)s and `Event`s. See [Training loop](/documents/docs/tutorials/training.md) and [`Events`](#) as a reference to phases and events.

## A guided example

There are 4 things you need to do to implement a custom callback:

1. Create a callback `struct` that subtypes [`Callback`](#)
2. Write event handlers with [`on`](#)
3. Define what state the callback accesses by implementing [`stateaccess`](#)
4. (Optionally) define dependencies on other callbacks with [`runafter`](#)

Let's go through them one at a time by implementing a simple callback that prints something after every batch.

### Callback `struct`

A callback definition has to subtype the abstract `Callback` type. It can include fields to use as internal state, but we don't need that here.

```julia
struct Printer <: Callback
end
```

### Event handlers

Now we need to add an event handler so that `Printer` can run some code when a step ends. Event handlers can be defined by adding a method to `FluxTraining.on`. It takes as arguments an [`Event`](#), a [`Phase`](#), the callback and the learner:

`on(event::Event, phase::Phase, callback::Callback, learner)`

The `event`, `phase` and `callback` are used to dispatch.

In this case, we want to run code at the end of a step, so the event we need to dispatch on is [`StepEnd`](#). We want it to run in any phase, so we use the abstract type `Phase`. The third argument type is the callback we want to add an event handler to. This gives us:

```julia
function FluxTraining.on(
        event::StepEnd,
        phase::Phase,
        printer::Printer,
        learner)
    println("Hello, World!")
end
```

We can now pass an instance of `Printer` when creating a `Learner` and the message will be printed at the end of every step.

### State

As seen above, the callback handler `on` receives as the last argument a `Learner` instance, allowing the callback to access and modify state. If we wanted to print the last step's loss instead of a generic message, we could update our definition of `on`:

```julia
function FluxTraining.on(
        event::EpochEnd,
        phase::Phase,
        printer::Printer,
        learner)
    println("Step loss:", learner.step.loss)
end
```
*(see [`Learner`](#) for in-depth documentation of the `Learner`'s state)*

The ability to modify any state is very powerful, but it can quickly become problematic when it is unclear which callbacks modify what state and what the correct order should be.
Because of that, *FluxTraining.jl* prevents callbacks from reading and modifying state by default. If we tried to use the above redefinition of `on`, we would get the following error:

```julia
FluxTraining.ProtectedException("Read access to Learner.step.loss disallowed.")
```

To fix that error, we need to implement `stateaccess`, a function that specifies what state a callback is allowed to read and write. In our case, we want to read the loss of the current step:

```julia
FluxTraining.stateaccess(::Printer) = (step = (loss = Read(),),)
```
*(see [`stateaccess`](#) for more information on how to implement it)*

After that definition, the above code will run fine. This might seem bothersome, but this extra information makes it possible to analyze state dependencies before any code is run and saves you from running into nasty, hard-to-find bugs that can occur when using many callbacks together.

### Dependencies

Let's improve our callback a bit by adding the current step number to the printed message, so it will look like this: `"Step 14 loss: 0.0032"`. For that we need to know what the current number of steps is. One way to go about this is to add a field to `Printer` that starts at `0` and is incremented every step.
Luckily, there already is a callback that tracks this kind of statistics, the [`Recorder`](#). It uses a special piece of state, `learner.cbstate`, to store a [`History`](#) with this information.

!!! info "Callback state"

    `learner.cbstate` is an object where callbacks can store state that they want to make available to other callbacks. Like any other piece of state, the callback writing to it needs to add a `Write()` permission to it using [`stateaccess`](#).

    What makes `cbstate` special is that when creating the callback graph, it is checked that every entry in `cbstate` that is accessed is being created first.

The update to the event handler looks like this:

```julia
function FluxTraining.on(
        event::EpochEnd,
        phase::Phase,
        printer::Printer,
        learner)
    step = learner.cbstate.history[phase].stepsepoch  # steps completed in current epoch
    println("Step ", , " loss:", learner.step.loss)
end
```

We also need to update the definition of `stateaccess` now:

```julia
FluxTraining.stateaccess(::Printer) = (
    step = (loss = Read(),),
    cbstate = (history = Read(),),
)
```

Since `Printer` depends on `Recorder` now, an error will be thrown if you try to use `Printer` without `Recorder`. And that's it, pass `Printer` to a `Learner` and test it out! The upside of jumping through some additional hoops is that using the callback in the wrong context will always result in an error, so the user can have peace of mind.

## Conflict resolution

When creating a `Learner`, a dependency graph is created. The graph is then analyzed to find possible conflicts (for example, when two callbacks update the same state). Conflicts are detected automatically and will result in an error. Conflicts happen when the same state is being modified by multiple callbacks and it is unclear which order of running them (if any) is valid.

### Resolving conflicts

There are two methods for resolving conflicts, `runafter` and `resolveconflict`.
`runafter` allows you to define list of callbacks that should run before the callback. For example, `Recorder` needs to run after all metrics:
```julia
FluxTraining.runafter(::Recorder) = (AbstractMetric,)
```

`resolveconflict` provides more granular control to handle a possible conflict between two callbacks. It takes two callbacks and defines how to resolve a conflict:

```julia
# the default, errors with a helpful message
resolveconflict(::C1, ::C2) = NotImplemented()    
# two callbacks can never be used together:
resolveconflict(::C1, ::C2) = Unresolvable()      
resolveconflict(::C1, ::C2) = NoConflict()        # there is no conflict, any run order is fine
resolveconflict(cb1::C1, cb2::C2) = RunFirst(cb1) # `cb1` must run before `cb2`.
                                                  # Equivalent to `runafter(::C2) = (C1,)
```

## Callback execution

By default, a topological ordering of the callbacks is created from the dependency graph and the callbacks are executed serially. This behavior can be overwritten with custom callback executors, for example to create a *Dagger.jl* node from the graph to allow callbacks to safely run in parallel where valid.
