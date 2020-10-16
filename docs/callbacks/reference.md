# Callback reference

## Included callbacks

*FluxTraining.jl* comes included with many callbacks. Some of them are added to `Learner` by default, here marked with a *.

### Metrics

- [`Loss`](#)*
- [`Metric`](#)

### Logging and I/O

- `Checkpointer`
- [`ProgressPrinter`](#)*
- [`Recorder`](#)*
- [`MetricsPrinter`](#)*

### Training loop

- [`EarlyStopping`](#)
- [`StopOnNaNLoss`](#)*
- [`Scheduler`](#)
- [`ToGPU`](#)

## API

The following types and functions can be used to create custom callbacks. Read the [custom callbacks guide](./custom.md) for more context.

- [`Callback`](#)
- [`stateaccess`](#)
- [`runafter`](#)
- [`resolveconflict`](#)