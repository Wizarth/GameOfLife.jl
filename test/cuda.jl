using CUDA
using Test

include("datasets.jl")

function test_glider_cuda()
    # Convert to a CUDA backed storage
    grid = Grid(
        CuMatrix(parent(glider))
    )

    for _ = 1:20
        grid = step_grid(grid)
    end

    # Convert to a CPU backed storage
    grid = Grid(
        collect(parent(grid))
    )

    # Gliders travel at 1/4 speed, so after 20 steps on a 5x5, should return to its origin
    @test glider == grid
end

test_glider_cuda()