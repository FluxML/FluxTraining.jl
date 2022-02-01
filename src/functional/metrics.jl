# FIXME: Metric(Accuracy) will give incorrect results and cannot be report in papers.
function accuracy(y_pred, y)
    y_pred_oc, y_oc = onecold(y_pred), onecold(y)
    match_count = sum(y_pred_oc .== y_oc)
    count = length(y_oc)
    if is_distributed_data_parallel()
        res = [match_count, count]
        MPIExtensions.Allreduce!(res, +, MPI.COMM_WORLD)
        return res[1] / res[2]
    else
        return match_count / count
    end
end
