using Test
using TestSetExtensions
using FluxTraining
using FluxTraining: EpochEnd, LR, getdataloader, getoptimparam, setoptimparam!, protect, Protected, ProtectedException, Read, Write
using Flux: trainable

include("./testdata.jl")
