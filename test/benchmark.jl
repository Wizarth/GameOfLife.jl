# Stand alone file that sets up some benchmark functions. Intended for use in the REPL.

using Pkg
Pkg.activate(".")

using GameOfLife

using BenchmarkTools
using CUDA

function random_grid(dims=(512,512))
    grid = Grid(rand(
        (Live, Dead, Dead, Dead, Dead),   # 1/5 chance of being Live
        dims...
    ))

    @benchmark step_grid($grid)
end

function random_grid_cuda(dims=(512,512))
    grid = Grid(
        CuMatrix(
            rand(
                (Live, Dead, Dead, Dead, Dead),   # 1/5 chance of being Live
                dims...
            )
        )
    )

    @benchmark CUDA.@sync step_grid($grid)
end

random_grid_cuda((8192,8192))