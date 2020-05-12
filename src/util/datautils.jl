

function splitdataset(f, dataset)
    trainidxs = filter(idx -> f(idx), 1:nobs(dataset))
    validxs = filter(idx -> !f(idx), 1:nobs(dataset))

    return datasubset(dataset, trainidxs), datasubset(dataset, validxs)
end

function splitdataset(dataset, splits::AbstractVector{Bool})
    splitdataset((idx) -> splits[idx], dataset)
end
