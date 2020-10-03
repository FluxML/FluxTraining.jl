# Callback reference

## Included callbacks

*FluxTraining.jl* comes included with many callbacks. Some of them are added to `Learner` by default, here marked with a *.

### Metrics

- [`Loss`](#)*
- [`Metric`](#)

### Logging and I/O

- [`Checkpointer`](#)
- [`ProgressBarLogger`](#)*
- [`Recorder`](#)*
- [`MetricsLogger`](#)*

### Training loop

- [`EarlyStopping`](#)
- [`StopOnNaNLoss`](#)*
- [`Scheduler`](#)
- [`ToGPU`](#)

Additionally, some abstract types are defined. It is recommended that you subtype from these where it makes sense so they will be play nicely with other callbacks.

- [`AbstractMetric`](#)
- [`AbstractLogger`](#) 

## API

The following types and functions can be used to create custom callbacks. Read the [custom callbacks guide](./custom.md) for more context.

- [`Callback`](#)
- [`stateaccess`](#)
- [`runafter`](#)
- [`resolveconflict`](#)