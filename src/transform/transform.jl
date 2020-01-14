

import Flux: onehot
using Images: Colorant, Gray, AbstractRGB, imresize, colorview, channelview, permuteddimsview


function normalize!(a, means, stds)
    for i = 1:3
        a[:,:,i] .-= means[i]
        a[:,:,i] ./= stds[i]
    end
    a
end
normalize(a, means, stds) = normalize!(copy(a), means, stds)

function denormalize!(a, means, stds)
    for i = 1:3
        a[:,:,i] .*= stds[i]
        a[:,:,i] .+= means[i]
    end
    a
end
denormalize(a, means, stds) = denormalize!(copy(a), means, stds)

imagetotensor(image::AbstractArray{<:AbstractRGB, 2}) = float.(permuteddimsview(channelview(image), (2, 3, 1)))

tensortoimage(tensor::AbstractArray{T, 3}) where T = colorview(RGB, permuteddimsview(tensor, (3, 1, 2)))
tensortoimage(tensor::AbstractArray{T, 2}) where T = colorview(Gray, permuteddimsview(tensor, (3, 1, 2)))

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


## Normalizing

struct Normalize
    means
    stds
end

(t::Normalize)(image) = normalize!(image, t.means, t.stds)



## Pipelining

function applytransform!(sample, f, argkeys, outputkeys)
    outputs = f((sample[arg] for arg in totuple(argkeys))...)

    for (output, outputkey) in zip(totuple(outputs), totuple(outputkeys))
        sample[outputkey] = output
    end

    return sample
end

totuple(x::Tuple) = x
totuple(x) = (x,)

# ToEltype

struct ToEltype{T} end
ToEltype(T::Type) = ToEltype{T}()

(t::ToEltype{T})(a::AbstractArray{<:T, 2}) where T = a
(t::ToEltype{T})(a::AbstractArray{U, 2}) where {T, U} = map(a) do x
    convert(T, x)
end

# OneHot

struct OneHot
    n
end
(t::OneHot)(x) = float(onehot(x, 1:t.n))


struct SampleTransform
    f
    argkeys
    outputkeys
end

(t::SampleTransform)(sample) = applytransform!(sample, t.f, t.argkeys, t.outputkeys)


struct Compose
    transforms
    Compose(xs...) = new(xs)
end

foldl
(t::Compose)(sample) = foldl((sample, f) -> f(sample), t.transforms; init = sample)
