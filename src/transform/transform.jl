
using Images: Colorant, imresize, channelview, permuteddimsview

include("./functional.jl")


function makedensitymaps(keypoints, size, sigma)
    dmaps = zeros(Float32, size..., length(keypoints))
    for (j, ks) in enumerate(keypoints)
        dmap = @view dmaps[:,:,j]
        densitymap!(dmap, ks, ones(length(ks)) * sigma; f = max)
    end
    dmaps
end


function normalize!(a, means, stds)
    for i = 1:3
        a[:,:,i] .-= means[i]
        a[:,:,i] ./= stds[i]
    end
    a
end

imageto3d(image) = float.(permuteddimsview(channelview(image), (2, 3, 1)))


# Transform structs

## RandomResizedCrop

struct RandomResizedCrop
    size::Tuple
    scales::Tuple
end


function getparams(t::RandomResizedCrop, oldsize)
    height, width = oldsize
    area = height * width
    for  _ = 1:10
        targetarea = rand(t.scales[1]:.01:t.scales[2]) * area
        h = w = Int(floor(sqrt(targetarea)))
        if 1 < w < width && 1 < h < height
            i = rand(1:height-h+1)
            j = rand(1:width-w+1)

            return i, j, h, w
        end
    end
    return 1, 1, height, width


end


function (t::RandomResizedCrop)(image::AbstractMatrix{<:Colorant})
    top, left, height, width = getparams(t, size(image))
    image = resizedcrop(image, top, left, height, width, t.size)
    return image
end


function (t::RandomResizedCrop)(
    image::AbstractMatrix{<:Colorant},
    poses::AbstractMatrix{<:Joint}
    )
    top, left, height, width = getparams(t, size(image))
    image = resizedcrop(image, top, left, height, width, t.size)
    poses = resizedcrop(poses, top, left, height, width, (t.size ./ (height, width)))
    return image, poses
end

## DownscalePoses

struct DownscalePoses
    factor::Integer
end

(t::DownscalePoses)(poses) = mapmaybe(k -> k ./ t.factor, poses)

## DensityMap

struct MakeDensityMap
    sigma
end

function(t::MakeDensityMap)(poses, height, width)
    keypoints = map(pose->filter(!isnothing, pose), eachcol(poses))
    return makedensitymaps(keypoints, (height, width), t.sigma)
end

## Normalizing

struct Normalize
    means
    stds
end

(t::Normalize)(image) = normalize!(image, t.means, t.stds)



## Pipelining

function applytransform!(sample::Dict, f, argkeys, outputkeys)
    outputs = f((sample[arg] for arg in totuple(argkeys))...)

    for (output, outputkey) in zip(totuple(outputs), totuple(outputkeys))
        sample[outputkey] = output
    end

    return sample
end

totuple(x::Tuple) = x
totuple(x) = (x,)


struct SampleTransform
    f
    argkeys
    outputkeys
end

(t::SampleTransform)(sample) = applytransform!(sample, t.f, t.argkeys, t.outputkeys)


struct Compose
    transforms
end
Compose(xs...) = Compose(xs)

(t::Compose)(sample) = foldl((sample, f) -> f(sample), t.transforms, init = sample)
