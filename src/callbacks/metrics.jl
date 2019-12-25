abstract type AbstractMetric <: AbstractCallback end

metricvalue(metric::T) where T<:AbstractMetric = throw("`metricvalue` not implemented for $(T)")

struct AverageMetric <: AbstractMetric
    fn
    value
    count
    AverageMetric(fn) = AverageMetric(fn, nothing, nothing)
end

Base.show(io::IO, metric::AverageMetric) = print(io, "average_$(metric.fn)")

function on_epoch_begin(metric::AverageMetric, state::TrainingState)
    metric.value = 0.
    metric.count = 0
end

function on_batch_end(metric::AverageMetric, state::TrainingState)
    metric.value += metric.fn(state.output, state.batchy)
    metric.count += 1
end

metricvalue(metric::AverageMetric) = metric.value / metric.count
