# Callback reference

## Included callbacks

*FluxTraining.jl* comes included with many callbacks. Some of them are added to `Learner` by default, here marked with a *.

### Logging and I/O

- [`Checkpointer`](#)
- [`Metrics`](#)*
- [`MetricsPrinter`](#)*
- [`ProgressPrinter`](#)*
- [`Recorder`](#)*
- [`SanityCheck`](#)*


### Training loop

- [`EarlyStopping`](#)
- [`StopOnNaNLoss`](#)*
- [`Scheduler`](#)
- [`ToGPU`](#)

## Utilities

There are also some callback utilities:

- [`CustomCallback`](#)
- [`throttle`](#)

## API

The following types and functions can be used to create custom callbacks. Read the [custom callbacks guide](./custom.md) for more context.

- [`Callback`](#)
- [`stateaccess`](#)
- [`runafter`](#)
- [`resolveconflict`](#)

