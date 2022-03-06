using CUDA
using StaticArrays

# We can't use CiruclarArray with a CuArray, it will bork in various ways

function kernel(in, out)
    gridDim = CUDA.gridDim()     # Number of blocks
    blockDim = CUDA.blockDim()   # Size of each block
    blockIdx = CUDA.blockIdx()   # Which block are we processing?
    threadIdx = CUDA.threadIdx() # Which cell of the block are we processing?

    # TODO: threadIdx x/y are Int32 . We probably should limit ourselves to Int32 rather than promote to Int64
    x = Int64(threadIdx.x + ((blockIdx.x-1) * blockDim.x))  # blockIdx is 1 indexed, which makes this math a little odd...
    y = Int64(threadIdx.y + ((blockIdx.y-1) * blockDim.y))

    # @cushow (x, y)
    
    # Get the neighbours of cellIdx
    # Perform wrapping
    # I feel like this is going to chew through performance, memory accesses are going
    # to be all over the place, from the viewpoint of the GPU
    # (max_x, max_y) = size(in)   # Could use gridDim instead of size here?
    max_x = Int64(gridDim.x * blockDim.x)
    max_y = Int64(gridDim.y * blockDim.y)

    # Bounds check x vs max_x in the case where the grid doesn't cleanly divide to the block size

    # @cushow (max_x, max_y)

    x_minus_1 = if x == 1
        max_x
    else
        x - 1
    end
    y_minus_1 = if y == 1
        max_y
    else
        y - 1
    end

    x_plus_1 = if x == max_x
        1
    else
        x + 1
    end

    y_plus_1 = if y == max_y
        1
    else
        y + 1
    end

    isLive(c) = c == Live ? 1 : 0

    num_live_neighbours = isLive(
        in[x_minus_1, y_minus_1]
    ) + isLive(
        in[x, y_minus_1]
    ) + isLive(
        in[x_plus_1, y_minus_1]
    ) + isLive(
        in[x_minus_1, y]
    ) + isLive(
        in[x_plus_1, y]
    ) + isLive(
        in[x_minus_1, y_plus_1]
    ) + isLive(
        in[x, y_plus_1]
    ) + isLive(
        in[x_plus_1, y_plus_1]
    )
    
    @inbounds out[x,y] = if in[x,y] == Live
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

function step_grid(cell_grid::CuArray{Cell})
    out = similar(cell_grid)

    # If threads not specified, it launches 1 thread
    # Maximum of 1024 threads (verify if this is constant etc)
    # So need to divide cell_grid into blocks as well

    threads = (16, 16)
    grid_size = size(cell_grid)
    # TODO: This should round up, currently rounds down
    blocks = (grid_size[1] ÷ 16, grid_size[2] ÷ 16)

    @cuda threads=threads blocks=blocks kernel(
        cell_grid, 
        out
        )

    return out
end