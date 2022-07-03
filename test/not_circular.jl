using Test

function test_notcircular()
    #=
        Create a 10x10 Matrix, then step the central 8x8 (2:9,2:9)

        This proves a CircularArray is not required, as long as there's neighbours that can be read
    =#
    grid = fill(Dead, 10, 10)

    @test step_grid(grid, CartesianIndices((2:9, 2:9))) == fill(Dead, 8, 8)
end

test_notcircular()