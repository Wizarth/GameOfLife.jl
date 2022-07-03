export step_grid, step_grid!

using StaticArrays: SVector

import Folds

"""
    Do not call this directly, call step_grid() or step_grid!() .

    Since this is internal, I haven't put type definitions in. Don't call it directly!
"""
@inline function _step_cell(cell_index, grid )
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

    if grid[cell_index] == Live
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
end

"""
This is more flexible than specifying Grid{Cell} directly for grid, as the implementation technically doesn't care what T is, only that it can be called against
==(Live). This is used to let it run on Grid{<:Real} .

In future, this might be even more flexible 
  * The neighbourhood could be programatically generated from the dims of grid and a distance range, making it able to accept 3d or more grids
    If done with a macro, and type templated on Val of range, it would still be fast
  * The Liveness check could be passed as an operator function
    ==() produces an invokable type (vs a generic function), so causes specialization and stays fast.
  * The function specifying the state change could also be abstracted
    This seems like it would impact performance?

Base.map(CartesianIndices) results in a Vector{Cell}, thus the need to reshape it to match grid.
Not sure what Folds.map resulted in. code_warntype reported it as Any.
Folds.map! requires the target to have the same dimensionality as CartesianIndices, so it's given a Matrix.

Folds.map! appears to be significantly faster, presumably because of dest_grid having a known type, and because the implementation is no longer needing to do work to determine what the target should be.
"""
function step_grid!(dest_grid::AbstractMatrix, grid::AbstractMatrix, coords::CartesianIndices{2})
    #=
        If step_grid! is called with a CircularMatrix as the destination, unwrap it as it's just
        adding overhead.
    =#
    if typeof(dest_grid) <: CircularArray
        error("This should be picked up by the specialization!")
    end

    @boundscheck checkbounds(Bool, grid, coords) || error("all coords must be inside grid")
    @boundscheck axes(dest_grid) == axes(coords) || error("dest_grid axis must match coords")

    #=
        There's an implicit convert between the result of _process_cell (Cell) and eltype(dest_grid) .
        I originally had a explicit convert in the body, but it slowed down performance.
        The type was being passed to the do body as an unknown DataType - i.e. the do body isn't
            specialized on the eltype of dest_grid, so the convert can't be folded out.
    =#
    Folds.map!(dest_grid, coords) do cell_index
        _step_cell(cell_index, grid)
    end
end

"""
    If step_grid! is called with a CircularMatrix as the destination, unwrap it as it's just
        adding overhead.
"""
function step_grid!(dest_grid::CircularArray{T, 2}, grid::AbstractMatrix, coords::CartesianIndices{2}) where {T}
    step_grid!(parent(dest_grid), grid, coords)
    return dest_grid
end

"""
    This implementation is for workijng on the CPU.
    There is a specialization for CircularArray of CuMatrix.
"""
Base.@propagate_inbounds function step_grid(grid::AbstractMatrix, coords::CartesianIndices{2})
    dest_grid = Matrix{eltype(grid)}(undef, size(coords))
    step_grid!(dest_grid, grid, coords)

    # Calling this constructor requires CircularArrays 1.3.1
    return typeof(grid)(dest_grid)
end

"""
Step an entire grid. Depends on CircularArray to perform looping around the edges.

CircularArray isn't strictly neccessary, except what other implementation can have offsets added/subtracted to every one of its CartesianIndices ?
"""
step_grid(grid::CircularArray{T,2}) where {T} = @inbounds step_grid(grid, CartesianIndices(grid))

"""
Step an entire grid. Depends on CircularArray to perform looping around the edges.

CircularArray isn't strictly neccessary, except what other implementation can have offsets added/subtracted to every one of its CartesianIndices ?
"""
function step_grid!(dest::CircularArray{T,2}, src::CircularArray{T,2}) where {T}
    @boundscheck axes(dest) == axes(src) || error("dest and src must have same axes")
    @inbounds step_grid!(dest, src, CartesianIndices(src))
end
