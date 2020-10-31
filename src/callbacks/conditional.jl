
# High-level interface

"""
    throttle(callback, event, freq = 1)
    throttle(callback, event, seconds = 1)

Throttle `event` for `callback` so that it is triggered either only every
`freq`'th time  or every `seconds` seconds.
"""
function throttle(callback, event; freq = nothing, seconds = nothing)
    xor(isnothing(freq), isnothing(seconds)) || error("Pass either `every` OR `seconds`.")
    if !isnothing(freq)
        return ConditionalCallback(callback, FrequencyThrottle(freq, event))
    elseif !isnothing(seconds)
        return ConditionalCallback(callback, TimeThrottle(seconds, event))
    end
end

# Implementation

abstract type CallbackCondition end

struct ConditionalCallback <: Callback
    callback::Callback
    condition::CallbackCondition
end

stateaccess(cc::ConditionalCallback) = stateaccess(cc.callback)

function on(event::Event, phase::Phase, cb::ConditionalCallback, learner)
    if shouldrun(cb.condition, event, phase)
        on(event, phase, cb.callback, learner)
    end
end


mutable struct FrequencyThrottle <: CallbackCondition
    freq
    event
    counter
    FrequencyThrottle(f, e, c = 1) = new(f, e, c)
end

function shouldrun(c::FrequencyThrottle, event, phase)
    if c.counter == 1
        c.counter = c.freq
        return true
    else
        c.counter -= 1
        return false
    end
end

mutable struct TimeThrottle <: CallbackCondition
    seconds
    event
    timer
    TimeThrottle(s, e, t = nothing) = new(s, e, t)
end


function shouldrun(c::TimeThrottle, event, phase)
    if isnothing(c.timer) || (time() - c.timer) > c.seconds
        c.timer = time()
        return true
    else
        return false
    end
end
