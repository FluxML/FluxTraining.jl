using ProgressMeter: Progress, next!
using Statistics: mean, std
using Images: RGB
using MLDataUtils: datasubset


function computestats(dataset; getimagefn = (sample) -> sample[:image])
    dataloader = DataLoader(
        dataset,
        8;
        collate_fn = identity,
        transform_fn = imagetotensor ∘ ToEltype(RGB) ∘ (sample -> sample[:image]))

    means = [0., 0., 0.]
    stds = [0., 0., 0.]

    p = Progress(nobs(dataloader.dataset), "Calculating image stats... ")
    for image in Training.makesamplechannel(dataloader)
        for ch in 1:3
            imagech = @view image[:,:,ch]
            means[ch] += mean(imagech)
            stds[ch] += std(imagech)
        end
        next!(p)
    end

    return means ./ nobs(dataset), stds ./ nobs(dataset)
end


function splitdataset(f, dataset)
    trainidxs = filter(idx -> f(idx), 1:nobs(dataset))
    validxs = filter(idx -> !f(idx), 1:nobs(dataset))

    return datasubset(dataset, trainidxs), datasubset(dataset, validxs)
end

function splitdataset(dataset, splits::AbstractVector{Bool})
    splitdataset((idx) -> splits[idx], dataset)
end
