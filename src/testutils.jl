# This file defines [`testlearner`](#) that constructs a [`Learner`](#)
# with a very simple optimisation problem. This should be used in tests
# for callbacks.


struct TestModel
    coeff
end
TestModel(coeff::Number) = TestModel([coeff])
Flux.trainable(m::TestModel) = (m.coeff,)
(m::TestModel)(x) = x .* m.coeff
Flux.@functor TestModel


function testbatch(batchsize, coeff)
    xs = rand(1.:100., batchsize)
    return (xs, xs .* coeff)
end


function testbatches(n::Int, coeff, batchsize = 8)
    (testbatch(batchsize, coeff) for _ ∈ 1:n)
end


"""
    testlearner(callbacks...[; opt, nbatches, coeff, batchsize, kwargs...])

Construct a [`Learner`](#) with a simple optimization problem. This
learner should be used in tests that require training a model, e.g.
for callbacks.
"""
function testlearner(
        args...;
        opt = Descent(0.001),
        nbatches = 16,
        coeff = 3,
        batchsize = 8,
        usedefaultcallbacks = false,
        kwargs...)
    model = TestModel(rand())
    data = collect(testbatches(nbatches, coeff, batchsize))
    Learner(
        model,
        (data, data),
        opt,
        Flux.mae,
        args...;
        usedefaultcallbacks = usedefaultcallbacks,
        kwargs...)
end
