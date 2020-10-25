# Training loop

The training loop, of course, is the core component in *FluxTraining.jl*.

Once you have a [`Learner`](#), the simplest way to start training is to run [`fit!`](#)`(learner, n)`. This will train for `n` epochs of one training phase and one validation phase each.

Like the [callback system](../callbacks/usage.md), the training loop has an extensible interface based on multiple dispatch.

The key abstraction are [`Phase`](#)s. In fact, the above `fit!(learner, n)` is just a shorthand for running `fit!(learner, [TrainingPhase(), ValidationPhase()])` `n` times. 

Each phase implements its own training logic which is run when calling `fit!(learner, phase)`:

- [`TrainingPhase`](#) does what you expect it to: it uses the training data iterator (`learner.data.training`), iterates over the batches, computes the forward and backward pass for each batch and updates the model parameters.
- [`ValidationPhase`](#) uses the validation data to compute the forward pass

If this looks like a regular old training loop, you're right! What makes the training loop customizable is the callback system. During training, [`Event`](#)s are thrown that [callbacks can hook into](../callbacks/custom.md).

Here's a rundown of the events thrown during a training and validation phase:



## Extending

Training and validation are currently the only two [`Phase`](#)s implemented, but the interface can be extended by implementing [`fitepochphase!`](#) and [`fitbatchphase!`](#). An example use case is implementing GAN training.


