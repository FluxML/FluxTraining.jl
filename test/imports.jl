using Test
using TestSetExtensions
using FluxTraining
using FluxTraining: EpochEnd, LearningRate, getdataiter, protect, Protected, ProtectedException
using Flux: trainable

include("./testdata.jl")
