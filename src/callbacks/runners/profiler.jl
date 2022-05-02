
# Data structures

struct TimingBetween
    phase::Any
    eventstart::Any
    eventend::Any
    timestart::Any
    timeend::Any
    duration::Any
end

TimingBetween(phase, es, ee, ts, te) = TimingBetween(phase, es, ee, ts, te, te - ts)

struct TimingCallback
    phase::Any
    cb::Any
    event::Any
    timestart::Any
    timeend::Any
    duration::Any
end

TimingCallback(phase, cb, e, ts, te) = TimingCallback(phase, cb, e, ts, te, te - ts)


# Runner

"""
    ProfileRunner() <: CallbackRunner

A profiling callback runner that measures times for callback
handlers and times between events. This allows for granular
benchmarking of any training loop.

## Examples

To use, pass as `cbrunner` argument to `Learner`:

```julia
cbrunner = ProfileRunner()
learner = Learner(model, data, opt, lossfn; cbrunner=cbrunner)
fit!(learner, 10)
```

After having trained, you can access the timings on fields:

- `cbrunner.timesbetween`: Stores timings between events
- `cbrunner.timescallbacks`: Stores timings for callback handlers
"""
mutable struct ProfileRunner <: FluxTraining.CallbackRunner
    df_fit::DataFrame
    df_cb::DataFrame
    _last::Any
end


ProfileRunner() = ProfileRunner(_new_df_fit(), _new_df_cb(), nothing)


function Base.show(io::IO, runner::ProfileRunner)
    print(io, "ProfileRunner(df_fit = ")
    summary(io, runner.df_fit)
    print(io, ", df_cb = ")
    summary(io, runner.df_cb)
    print(io, ")")
end


_new_df_fit() = DataFrame(
    phase = PooledArray(Type{<:Phase}[], UInt8),
    eventstart = PooledArray(Type{<:Event}[], UInt8),
    eventend = PooledArray(Type{<:Event}[], UInt8),
    timestart = Float64[],
    timeend = Float64[],
)

_new_df_cb() = DataFrame(
    phase = PooledArray(Type{<:Phase}[], UInt8),
    event = PooledArray(Type{<:Event}[], UInt8),
    callback = FluxTraining.Callback[],
    timestart = Float64[],
    timeend = Float64[],
)



function FluxTraining.handle(
    runner::ProfileRunner,
    event::E,
    phase::P,
    learner,
) where {E<:Event,P<:Phase}
    # add timing for inbetween
    last = runner._last
    if last !== nothing
        timeend = Zygote.ignore(() -> Base.time())
        lastevent, lasttime, lastphase = last
        if lastphase == P
            Zygote.ignore() do
                push!(
                    runner.df_fit,
                    (;
                        phase = P,
                        eventstart = lastevent,
                        eventend = E,
                        timestart = lasttime,
                        timeend = timeend,
                    ),
                )
            end
        end
    end

    # execute callback and add timing for it
    idxs = Zygote.ignore() do
        LightGraphs.topological_sort_by_dfs(learner.callbacks.graph)
    end
    for i in idxs
        cb = learner.callbacks.cbs[i]
        timestart = Zygote.ignore(() -> Base.time())
        FluxTraining._on(event, phase, cb, learner)
        Zygote.ignore() do
            timeend = Base.time()
            push!(
                runner.df_cb,
                (;
                    phase = P,
                    event = E,
                    callback = cb,
                    timestart = timestart,
                    timeend = timeend,
                ),
            )
        end
    end

    # update `last` so next between time can be measured
    runner._last = (E, Zygote.ignore(() -> Base.time()), P)
    return nothing
end


# ### Data transformations
#
# Get the data into a usable shape for further analysis.

"""
    getsteptimings(profilerunner[, Phase]) -> GroupedDataFrame

Group the data of step timings by the events that they occur between.
"""
function getsteptimings(runner::ProfileRunner, P = AbstractTrainingPhase)
    return groupby(
        subset(
            combine(
                runner.df_fit,
                [:timeend, :timestart] => ((e, s) -> e - s) => :duration,
                :phase,
                :eventstart,
                :eventend,
            ),
            :phase => (ps -> ps .<: P),
            :eventstart => (es -> ((es .!= EpochBegin) .& (es .!= EpochEnd))),
            :eventend => (es -> ((es .!= EpochBegin) .& (es .!= EpochEnd))),
        ),
        [:eventstart, :eventend],
    )
end

# ### Analysis and visualization
#
# Provide helpful analyses that show most important timings and help with
# benchmarking and identifying bottlenecks.

"""
    showsteptimings(profilerunner)
    showsteptimings(io, profilerunner, P = AbstractTrainingPhase; metrics = [...])


"""
function showsteptimings(
    io::IO,
    runner::ProfileRunner,
    P = AbstractTrainingPhase;
    metrics = [median, minimum, maximum],
)
    gdf = getsteptimings(runner, P)
    rownames = ["$(k.eventstart) => $(k.eventend)" for k in keys(gdf)]
    rowdata = [metricfn(eventdf.duration .* 1000) for eventdf in gdf, metricfn in metrics]
    pretty_table(
        io,
        rowdata,
        header = (string.(metrics), repeat(["ms"], length(metrics))),
        row_names = rownames,
        row_name_column_title = "Event",
        highlighters = _timinghighlighter(),
        formatters = ft_printf("%5.3f"),
    )
end
showsteptimings(args...; kwargs...) = showsteptimings(stdout, args...; kwargs...)


# #### PrettyTables.jl utilities

_timinghighlighter() = Highlighter(
    (data, i, j) -> true,
    function (h, data, i, j)
        ext = extrema(data[:, j])
        ext = 0., ext[2]
        return Crayon(
            background = _cvtcolor(
                get(
                    ColorScheme(range(colorant"black", colorant"darkorange4")),
                    data[i, j],
                    ext,
                ),
            ),
            foreground = _cvtcolor(
                get(
                    ColorScheme(range(colorant"gray", colorant"white")),
                    data[i, j],
                    ext,
                ),
            ),
        )
    end,
)

_cvtcolor(c::Color) = (
    round(Int, Colors.red(c) * 255),
    round(Int, Colors.green(c) * 255),
    round(Int, Colors.blue(c) * 255),
)
