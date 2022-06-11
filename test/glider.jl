using Test

function test_glider()
    grid = glider

    for _ = 1:20
        grid = step_grid(grid)
    end

    # Gliders travel at 1/4 speed, so after 20 steps on a 5x5, should return to its origin
    @test glider == grid
end

test_glider()