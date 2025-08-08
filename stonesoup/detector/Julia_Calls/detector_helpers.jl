#=
detector_helpers:
- Julia version: 
- Author: mattb242
- Date: 2025-08-08
=#

module JuliaBayes

using WidebandDoA
using Statistics
using Random

function detect_mixture(rng, sample, stats; burn_fraction = 0.1)
    burned = samples[Int(end*burn_fraction):end]

    if isempty(stats)
        return nothing, nothing
    end

    k_mixture = quantile([stat.order for stat in stats], 0.9 |> Base.Fix1(round,Int))
    φ_post = [[target.phi for target in sample] for sample in burned]
    mixture, labels = WidebandDoA.relabel(rng, φ_post, k_mixture; show_progress = false)

    return mixture, labels

end

end
