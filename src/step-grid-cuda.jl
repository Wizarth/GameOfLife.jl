export step_grid_cuda

using StaticArrays: SVector

using CUDA
using Adapt

"""
    This is not exported for a reason, do not use directly!
"""
function _cuda_kernel(grid, out)
    blockDim = CUDA.blockDim()   # Size of each block
    blockIdx = CUDA.blockIdx()   # Which block are we processing?
    threadIdx = CUDA.threadIdx() # Which cell of the block are we processing?

    x = threadIdx.x + ((blockIdx.x-1) * blockDim.x)  # blockIdx is 1 indexed, which makes this math a little odd...
    y = threadIdx.y + ((blockIdx.y-1) * blockDim.y)

    # Prevent processing a cell_index outside of CartesianIndices(grid)
    if x > size(grid, 1) || y > size(grid, 2)
        return nothing
    end

    cell_index = CartesianIndex((x, y))

    neighbours = SVector(
        # This does hard code the nature of the neighbourhood here
        # If we wanted to make this more generic, we'd have to use a macro
        grid[cell_index + CartesianIndex(-1, -1)],
        grid[cell_index + CartesianIndex(0, -1)],
        grid[cell_index + CartesianIndex(1, -1)],
        grid[cell_index + CartesianIndex(-1, 0)],
        grid[cell_index + CartesianIndex(1, 0)],
        grid[cell_index + CartesianIndex(-1, 1)],
        grid[cell_index + CartesianIndex(0, 1)],
        grid[cell_index + CartesianIndex(1, 1)]
    )
    num_live_neighbours = count(==(Live), neighbours)

    @inbounds out[cell_index] = if grid[cell_index] == Live
        if num_live_neighbours < 2
            Dead
        # elseif num_live_neighbours == 2 || num_live_neighbours == 3
        elseif num_live_neighbours < 4
            Live
        else
            Dead
        end
    else
        if num_live_neighbours == 3
            Live
        else
            Dead
        end
    end

    return nothing
end

"""

"""
function step_grid(grid::CircularArray{T,2,ArrayT}) where {T, ArrayT<:CuMatrix}
    # new_grid is a CuMatrix, not a CircularMatrix, as we dont need the circular nature when writing
    new_grid = similar(parent(grid))

    #= This works

    threads = (16, 16)
    grid_size = size(grid)
    blocks = (cld(grid_size[1], threads[1]), cld(grid_size[2], threads[2]))

    #=
    @device_code_warntype @cuda threads=threads blocks=blocks _cuda_kernel(
        grid, 
        new_grid
    )
    =#
    @cuda threads=threads blocks=blocks _cuda_kernel(
        grid, 
        new_grid
    )

    typeof(grid)(new_grid)
    =#

    # Pull out the smarts from @cuda, to let us work out an optimal thread count 

    device_grid = cudaconvert(grid)
    device_new_grid = cudaconvert(new_grid)

    kernel = cufunction(cudaconvert(_cuda_kernel), Tuple{Core.Typeof(device_grid), Core.Typeof(device_new_grid)})

    maxthreads = CUDA.maxthreads(kernel)
    grid_size = size(grid)
    threads_x = min(maxthreads, grid_size[1])
    threads_y = min(maxthreads ÷ threads_x, grid_size[2])

    threads = (threads_x, threads_y)
    blocks = (cld(grid_size[1], threads[1]), cld(grid_size[2], threads[2]))

    CUDA.call( kernel, device_grid, device_new_grid; threads=threads, blocks=blocks)

    typeof(grid)(new_grid)
end

Adapt.@adapt_structure CircularArray