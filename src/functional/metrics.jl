
function accuracy(y_pred, y)
    return mean(onecold(y_pred) .== onecold(y))
end
