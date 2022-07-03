using Test

function test_glider()
    grid = glider

    for _ = 1:20
        grid = step_grid(grid)
    end

    # Gliders travel at 1/4 speed, so after 20 steps on a 5x5, should return to its origin
    @test glider == grid
end

"""
    For machine learning purposes, it would be useful to process a float grid as
    if it were Cells
"""
function test_glider_float()
    # Convert to a float grid
    grid = Grid{Float32}(glider)

    for _ = 1:20
        grid = step_grid(grid)
    end

    # Convert to a Cell grid
    grid = Grid{Cell}(grid)

    # Gliders travel at 1/4 speed, so after 20 steps on a 5x5, should return to its origin
    @test glider == grid
end

"""
    Test using step_grid! directly with an existing grid
"""
function test_glider!()
    grid_src = copy(glider)
    grid_dest = similar(glider)

    for _ = 1:20
        step_grid!(grid_dest, grid_src)
        grid_src, grid_dest = grid_dest, grid_src
    end

    # Gliders travel at 1/4 speed, so after 20 steps on a 5x5, should return to its origin
    @test glider == grid_src
end

test_glider()
test_glider_float()
test_glider!()