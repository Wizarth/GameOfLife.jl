export step_grid

using StaticArrays: SVector

"""
CircularArray isn't strictly neccessary, except what other implementation can have offsets added/subtracted to every one of its CartesianIndices ?

This is more flexible than specifying Grid directly, as the implementation technically doesn't care what T is, only that it can be called against
==(Live)

In future, this might be even more flexible 
  * The neighbourhood could be programatically generated from the dims of grid and a distance range, making it able to accept 3d or more grids
    If done with a macro, and type templated on Val of range, it would still be fast
  * The Liveness check could be passed as an operator function
    ==() produces an invokable type (vs a generic function), so causes specialization and stays fast.
  * The function specifying the state change could also be abstracted
    This seems like it would impact performance?
"""
function step_grid(grid::CircularArray{T,2}) where {T}
    new_grid = map(CartesianIndices(grid)) do cell_index
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

    # Calling this constructor requires CircularArrays 1.3.1
    return typeof(grid)(reshape(new_grid, size(grid)))
end