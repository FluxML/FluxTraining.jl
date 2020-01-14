
using Images: Colorant, imresize


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


function resize(image::AbstractMatrix{T}, size) where T<:Colorant
    return imresize(image, size...)
end


function resizedcrop(image::AbstractArray{T}, top, left, height, width, size) where T<:Colorant
    image = crop(image, top, left, height, width)
    image = resize(image, size)
    return image
end
