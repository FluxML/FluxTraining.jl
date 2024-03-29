
# News

## Unreleased

### Added

- Support for [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) https://github.com/FluxML/FluxTraining.jl/pull/114.

## [0.3.0] - 04.04.2022

### Added

- Short-hand `Learner` method `Learner(model, lossfn[; data, optim, callbacks, kwargs...])`. The old method still exists but use is discouraged.
- Compatibility with Flux.jl v0.13 once released 
- [Pollen.jl doucmentation frontend](https://fluxml.ai/FluxTraining.jl/dev/i/?id=documents%2FREADME.md)

### Changed

- (BREAKING) Hyperparameter scheduling now uses ParameterSchedules.jl instead of Animations.jl for defining schedules. See [the docs](https://fluxml.ai/FluxTraining.jl/dev/documents/docs/tutorials/hyperparameters.md/).
- Fixes on documentation pages

## [0.2.0]

### Added

- New training loop API that is easier to extend. Defining a `Phase` and `step!` is all you need. See [the new tutorial](docs/tutorials/training.md) and [the new reference](docs/reference/training.md).
  - Relevant functions: [`epoch!`](#), [`step!`](#), [`runstep`](#), [`runepoch`](#)
- Added `CHANGELOG.md` (this file)
- [`AbstractValidationPhase`](#) as supertype for validation phases
- Documentation for callback helpers on [reference page](docs/callbacks/reference.md)

### Changed

- `Batch*` renamed to `Step*`:
  - events: `BatchBegin` now `StepBegin`, `BatchEnd` now `StepEnd`
  - `CancelBatchException` now `CancelStepException`.
  - field `Learner.batch` now `Learner.step`
- `Learner.step/batch` is no longer a special `struct` but now a `PropDict`, allowing you to set arbitrary fields. 
- `Learner.model` can now be a `NamedTuple/Tuple` of models for use in custom training loops. Likewise, `learner.params` now resembles the structure of `learner.model`, allowing separate access to parameters of different models.
- Callbacks
  - Added [`init!`](#) method for callback initilization, replacing the `Init` event which required a `Phase` to implement.
  - [`Scheduler`](#) now has internal step counter and no longer relies on `Recorder`'s history. This makes it easier to replace the scheduler without needing to offset the new schedules.
  - [`EarlyStopping`](#) callback now uses criteria from [EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl)

### Removed

- Removed old training API. Methods `fitbatch!`, `fitbatchphase!`, `fitepoch!`, `fitepochphase!` have all been removed.
