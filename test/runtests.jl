include("./datasets.jl")

using Test

@testset "Glider" begin include("./glider.jl") end
@testset "CUDA" begin include("./cuda.jl") end
@testset "show" begin include("./show.jl") end
@testset "GridPermutations" begin include("./permutations.jl") end
@testset "Not Circuclar" begin include("./not_circular.jl") end
