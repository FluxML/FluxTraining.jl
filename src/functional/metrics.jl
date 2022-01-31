# FIXME: Metric(Accuracy) will give incorrect results and cannot be report in papers.
function accuracy(y_pred, y)
    matches = onecold(y_pred) .== onecold(y)
    match_count = sum(matches)
    count = length(y)
    if is_distributed_data_parallel()
        res = [match_count, count]
        MPIExtensions.Allreduce!(res, +, MPI.COMM_WORLD)
        return res[1] / res[2]
    else
        return match_count / count
    end
end
