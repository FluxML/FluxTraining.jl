# Training loop API reference

The training loop API centers around the abstract type [`Phase`](#) and the function [`step!`](#). To [implement a custom training](../tutorials/training.md), you need to

## Usage

- fit for `n` epochs of supervised training and validation using [`fit!`](#)`(learner, n)`
- train for an epoch using [`epoch!`](#)`(learner, phase, dataiter)`


## Extending

- subtype [`Phase`](#)
- implement [`step!`](#) 

You can optionally

- overwrite default [`epoch!`](#) implementation
- implement [`phasedataiter`](#) to define which data iterator should be used when `epoch!` is called without one.
- create custom [`Callback`](#) and [`Event`](#)s with event handlers that dispatch on your `Phase` subtype.

### Control flow

Inside callback handlers and `step!` implementations, you can throw [`CancelFittingException`](#) to stop the training and [`CancelEpochException`](#) and [`CancelStepException`](#) to skip the current epoch or step.