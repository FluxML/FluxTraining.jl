import OnlineStats

function smoothed(values::AbstractVector{T}, β = 0.98) where T
    m = Mean(T, weight = OnlineStats.ExponentialWeight(β))
    return [OnlineStats.value(OnlineStats.fit!(m, v)) for v in values]
end
