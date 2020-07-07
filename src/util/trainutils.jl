
function getbatch(dataloader::DataLoader, batchsize = dataloader.batchsize; offset = 0)
    idxs = 1 + offset:offset + batchsize
    samples = [DataLoaders.getsample(dataloader, idx) for idx in idxs]
    batch = DataLoaders.collate(samples)

    return batch
end
getbatch(databunch::DataBunch, train = true; kwargs...) = getbatch(train ? databunch.traindl : databunch.valdl; kwargs...)
getbatch(learner::Learner, train = true; kwargs...) = getbatch(learner.databunch, train; kwargs...)


function getoutputs(model, batch)
    x, y = gpu(batch)
    y_pred = gpu(model)(x)

    return y_pred, y
end

getoutputs(learner::Learner, batch = getbatch(learner)) = getoutputs(learner.model, batch)


function getloss(learner, batch = getbatch(learner))
    y_pred, y = getoutputs(learner, batch)
    learner.lossfn(y_pred, y)
end
