using Plots: plot, plot!

function plotlrfinder(recorder::Recorder, β = 0.02)
    lrs = recorder.stepstats[LR]
    losses = recorder.stepstats["Loss"]
    plot(lrs, losses, xscale=:log10)
end


function plotschedule(scheduledict::Dict)
    epochs = maximum([Training.duration(sched) for sched in values(scheduledict)])
    p = plot()
    for (P, sched) in scheduledict
        xs, ys = sampleschedule(sched, epochs = epochs)
        plot!(p, xs, ys, label=string(P))
    end

    return p
end
plotschedule(scheduler::ParamScheduler) = plotschedule(scheduler.scheduledict)


function plotlosses(losses, β = 0.02)
    plot(smoothed(losses, β), ylim = (0, maximum(losses)))
end
