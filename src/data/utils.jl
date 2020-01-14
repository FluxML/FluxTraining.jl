import ProgressMeter: Progress, next!
import Statistics: mean, std
import Images: RGB


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
