using Test
using GameOfLife

function test_permutation()
    # This isn't really exhaustive, is it? But it would catch every issue I encountered during development

    glider_perm = GameOfLife.GridPermutation(glider)
    
    @test glider == Grid(glider_perm)
    # This isn't an exact equality (glider is a Grid, collect() returns a Matrix{Cell}) but it's also good enough
    @test glider == collect(glider_perm)

    @test GameOfLife.max_permutation(glider) == 2 ^ prod(size(glider))
end

test_permutation()