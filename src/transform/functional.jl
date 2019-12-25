
using Images


function crop(image::AbstractArray{T, 2},  top, left, height, width) where T<:Colorant
    h, w = size(image)

    # pad if necessary
    if (h < (top + height)) || (w < (left + width))
        tocrop = zeros(T, max(h, top + height), max(w, left + width))
        tocrop[1:h, 1:w] .= image
    else
        tocrop = image
    end

    return tocrop[top:top+height-1, left:left+width-1]
end


function crop(poses::AbstractMatrix{<:Joint}, top, left, height, width)
    mapmaybe(poses) do keypoint
        y, x = keypoint
        if (top <= y <= top+height-1) && (left <= x <= left+width-1)
            # translate by upper corner of crop
            return (y - top + 1, x - left + 1)
        else
            return nothing
        end
    end
end

function resize(image::AbstractMatrix{T}, size) where T<:Colorant
    return imresize(image, size...)
end

function resize(poses::AbstractArray{<:Joint}, factors)
    mapmaybe(poses) do keypoint
        keypoint .* factors
    end
end



function resizedcrop(image::AbstractArray{T}, top, left, height, width, size) where T<:Colorant
    image = crop(image, top, left, height, width)
    image = resize(image, size)
    return image
end

function resizedcrop(
    poses::AbstractArray{<:Joint},
    top,
    left,
    height,
    width,
    resizefactors) where T<:Colorant
    poses = crop(poses, top, left, height, width)
    poses = resize(poses, resizefactors)
    return poses
end


# utils

mapmaybe(f, a) = [!isnothing(x) ? f(x) : x for x in a]
