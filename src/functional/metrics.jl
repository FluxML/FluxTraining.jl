
function accuracy(y_pred, y)
    return mean(onecold(cpu(softmax(y_pred))) .== onecold(cpu(y)))
end
