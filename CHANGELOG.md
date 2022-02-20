
# News

## [0.2.0] – Unreleased

This is a **breaking** release.

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