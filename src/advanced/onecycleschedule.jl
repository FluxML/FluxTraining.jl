function onecycleschedule(
        nepochs,
        maxlr;
        startfactor = 1/25,
        endfactor = 1/1e5,
        pctstart = 0.25)::Vector{ParamSchedule}

    upschedule = ParamSchedule(
        nepochs * pctstart,
        maxlr * startfactor,
        maxlr,
        anneal_cosine)
    downschedule = ParamSchedule(
        nepochs * (1-pctstart),
        maxlr,
        maxlr * endfactor,
        anneal_cosine)

    return [upschedule, downschedule]
end
