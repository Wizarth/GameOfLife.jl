export step_grid_cuda

using StaticArrays: SVector

using CUDA
using Adapt

"""
    This is not exported for a reason, do not use directly!

    TODO: Parameter ordering is non-Julia-n. out should be first.
"""
function _cuda_kernel(grid, out, coords)
    blockDim = CUDA.blockDim()   # Size of each block
    blockIdx = CUDA.blockIdx()   # Which block are we processing?
    threadIdx = CUDA.threadIdx() # Which cell of the block are we processing?

    x = threadIdx.x + ((blockIdx.x-1) * blockDim.x)  # blockIdx is 1 indexed, which makes this math a little odd...
    y = threadIdx.y + ((blockIdx.y-1) * blockDim.y)

    # Prevent processing a cell_index outside of CartesianIndices(grid)
    if x > size(grid, 1) || y > size(grid, 2)
        return nothing
    end

    cell_index = coords[x, y]

    @inbounds out[cell_index] = _step_cell(cell_index, grid)

    return nothing
end

"""

"""
function step_grid!(new_grid::ArrayT, grid::CircularArray{T,2,ArrayT}, coords::CartesianIndices{2}) where {T, ArrayT<:CuMatrix}    
    @boundscheck checkbounds(Bool, grid, coords) || error("all coords must be inside grid")
    @boundscheck axes(new_grid) == axes(coords) || error("new_grid axis must match coords")


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
    device_coords = cudaconvert(coords)

    kernel = cufunction(cudaconvert(_cuda_kernel), Tuple{Core.Typeof(device_grid), Core.Typeof(device_new_grid), Core.Typeof(device_coords)})

    maxthreads = CUDA.maxthreads(kernel)
    grid_size = size(coords)
    threads_x = min(maxthreads, grid_size[1])
    threads_y = min(maxthreads ÷ threads_x, grid_size[2])

    threads = (threads_x, threads_y)
    blocks = (cld(grid_size[1], threads[1]), cld(grid_size[2], threads[2]))

    CUDA.call( kernel, device_grid, device_new_grid, device_coords; threads=threads, blocks=blocks)
end

Base.@propagate_inbounds function step_grid(grid::CircularArray{T,2,ArrayT}, coords::CartesianIndices{2}) where {T, ArrayT<:CuMatrix{T}}
    # new_grid is a CuMatrix, not a CircularMatrix, as we dont need the circular nature when writing
    new_grid = similar(parent(grid))
    step_grid!(new_grid, grid, coords)

    # Calling this constructor requires CircularArrays 1.3.1
    typeof(grid)(new_grid)
end

Adapt.@adapt_structure CircularArray