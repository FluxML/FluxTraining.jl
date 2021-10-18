
# Data structures

struct TimingBetween
    phase
    eventstart
    eventend
    timestart
    timeend
    duration
end

TimingBetween(phase, es, ee, ts, te) = TimingBetween(phase, es, ee, ts, te, te-ts)

struct TimingCallback
    phase
    cb
    event
    timestart
    timeend
    duration
end

TimingCallback(phase, cb, e, ts, te) = TimingCallback(phase, cb, e, ts, te, te-ts)


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
    timesbetween
    timescallbacks
    last
end

ProfileRunner() = ProfileRunner(
    StructArray{TimingBetween}(
        phase = Phase[],
        eventstart = Type{<:Event}[],
        eventend = Type{<:Event}[],
        timestart = Float64[],
        timeend = Float64[],
        duration = Float64[],
    ),
    StructArray{TimingCallback}(
        phase = Phase[],
        cb = FluxTraining.Callback[],
        event = Type{<:Event}[],
        timestart = Float64[],
        timeend = Float64[],
        duration = Float64[],
    ),
    nothing
)


function FluxTraining.handle(runner::ProfileRunner, event, phase, learner)
    # add timing for inbetween
    last = runner.last
    if last !== nothing
        t = Zygote.ignore(() -> Base.time())
        lastevent, lasttime, lastphase = last
        if lastphase == phase
            Zygote.ignore() do
                timing = TimingBetween(
                    phase, typeof(lastevent), typeof(event), lasttime, t)
                push!(runner.timesbetween, timing)
            end
        end
    end

    # execute callback and add timing for it
    idxs = Zygote.ignore() do
        LightGraphs.topological_sort_by_dfs(learner.callbacks.graph)
    end
    for i in idxs
		cb = learner.callbacks.cbs[i]
		starttime = Zygote.ignore(() -> Base.time())
        FluxTraining._on(event, phase, cb, learner)
		Zygote.ignore() do
            timing = TimingCallback(phase, cb, typeof(event), starttime, Base.time())
            push!(runner.timescallbacks, timing)
		end
    end

    # update `last` so next between time can be measured
    runner.last = (event, Zygote.ignore(() -> Base.time()), phase)
    nothing
end


# Analysis

# TODO
