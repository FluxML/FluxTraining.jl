using DataLoaders
using MLDataUtils: splitobs
using FluxTraining
import Flux: trainable


function dummydata(n, factor)
    xs = collect(1:n)
    ys = xs .* factor
    return (xs, ys)
end

function dummydatabunch(n, factor; batchsize = 8)
    ds = dummydata(100, factor)
    trainds, valds = splitobs(ds)
    return DataBunch(
        DataLoader(trainds, batchsize),
        DataLoader(valds, batchsize),
    )
end

struct DummyModel
    factor
end
Flux.@functor DummyModel
Flux.trainable(dm::DummyModel) = [dm.factor]
(m::DummyModel)(x) = x .* m.factor


function dummylearner(factor; learnerkwargs...)
    return Learner(
        DummyModel([rand()]),
        dummydatabunch(128, factor),
        Descent(0.0001),
        Flux.mse;
        device = cpu,
        learnerkwargs...
    )
end
