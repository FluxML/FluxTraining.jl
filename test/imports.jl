using Test
using TestSetExtensions
using FluxTraining
using Animations
using Colors
using FluxTraining: EpochEnd, LearningRate, protect, Protected,
    ProtectedException, Read, Write, Events, Phases, runstep, runepoch, epoch!, step!

import FluxTraining.Phases: Phase
using Flux: trainable
using Suppressor
using FluxTraining: CHECKS, SanityCheckException
using Zygote

include("./testdata.jl");
