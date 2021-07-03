# Callback reference

## Included callbacks

*FluxTraining.jl* comes included with many callbacks. Some of them are added to `Learner` by default, here marked with a *.

| **Callback**            | **Description**                                                     |
| ----------------------- | ------------------------------------------------------------------- |
| [`Metrics`](#)*         | Tracks loss and additional metrics on a per-step and per-epoch base |
| [`Recorder`](#)*        | Records training stats like number of steps and epochs              |
| [`ProgressPrinter`](#)* | Prints a progress bar for the current epoch during training         |
| [`MetricsPrinter`](#)*  | Prints out metrics after every epoch                                |
| [`SanityCheck`](#)*     | Performs sanity checks on data, model and loss before training      |
| [`StopOnNaNLoss`](#)    | Stops training early if a step loss is `NaN`                        |
| [`ToGPU`](#)            | Trains using a CUDA GPU if available                                |
| [`Checkpointer`](#)     | Saves the model after every epoch                                   |
| [`EarlyStopping`](#)    | Stops training early when a criterion is met                        |
| [`Scheduler`](#)        | Schedules hyperparameters                                           |
| [`LogMetrics`](#)       | Logs metrics to a logging backend                                   |
| [`LogHyperParams`](#)   | Logs hyperparameters to a logging backend                           |
| [`LogVisualization`](#) | Logs visualization to a logging backend                             |
| [`LogHistograms`](#)    | Logs model weight histograms to a logging backend                   |


There are also some utilities for creating callbacks:

- [`CustomCallback`](#) to quickly hook a function into an event
- [`throttle`](#) to run a callback only after every `n` events or every `t` seconds

And for working with callbacks on an existing [`Learner`](#):

- [`setcallbacks!`](#)
- [`addcallback!`](#)
- [`getcallback`](#)
- [`replacecallback!`](#)
- [`removecallback!`](#)


## Extension API

The following types and functions can be used to create custom callbacks. Read the [custom callbacks guide](./custom.md) for more context.

- [`Callback`](#)
- [`stateaccess`](#)
- [`runafter`](#)
- [`resolveconflict`](#)

