# Stand alone file that sets up some benchmark functions. Intended for use in the REPL.

using GameOfLife

using BenchmarkTools

function random_grid(dims=(512,512))
    grid = Grid(rand(
        (Live, Dead, Dead, Dead, Dead),   # 1/5 chance of being Live
        dims...
    ))

    @benchmark step_grid($grid)
end