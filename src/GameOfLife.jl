module GameOfLife

using CircularArrays
using Luxor

displaySize = 10
compactDisplaySize = 2

@enum Cell::UInt8 Live=true Dead=false

using Base: show, convert

Base.convert(::Type{Bool}, cell::Cell) = cell == Live

function Base.show(io::IO, ::MIME"text/html", v::Cell)
	displayWidth, displayHeight = get(io, :displaysize, (3, 3))
	style = string("background: ", v == Live ? "black; " : "white; ")
	compact = displayWidth < 3 || displayHeight < 3
	if compact
		style = string(
			style,
			"height: $(displayHeight)px; width: $(displayWidth)px; "
		)
	else
		style = string(
			style,
			"height: $(displayHeight)px; width: $(displayWidth)px; ",
			"border: 1px solid ",
			v == Live ? "white; " : "black; "
		)
	end
	write(io, """<span style="$(style) display: inline-block;"></span>""")
end

function Base.show(io::IO, mime::MIME"text/html", vec::AbstractVector{Cell})
	(displayWidth, displayHeight) = get(io, :displaysize, (100, 10))
	cellWidth = max(1, displayWidth ÷ size(vec, 1))
	displayHeight = min(displayHeight, cellWidth)
	cellIO = IOContext(io, :displaysize => (cellWidth, displayHeight))

	write(io, """<div style="height: $(displayHeight)px; width:$(displayWidth)px;">""")
	for cell in vec
		Base.show(cellIO, mime, cell)
	end
	write(io, """</div>""")
end

function Base.show(io::IO, mime::MIME"text/html", cell_grid::AbstractMatrix{Cell})
	displayHeight, displayWidth = get(io, :displaysize, (600,600))
	cellHeight = displayHeight / size(cell_grid, 1)
	cellWidth = displayHeight / size(cell_grid, 2)
	
	# Calculate a border size
	borderSize = 0.5
	if min(cellHeight / 10, cellWidth / 10) < borderSize
		borderSize = 0
	end
	
	write(io, """<svg width="$(displayWidth)" height="$(displayHeight)">
		<style>
			rect {
				stroke-width: $(borderSize)px;
			}
			.dead {
				fill: white;
				stroke: black;
			}
			.live {
				fill: black;
				stroke: white;
			}
		</style>
		""")
	
	for y in 1:size(cell_grid, 1)
		for x in 1:size(cell_grid, 2)
			@inbounds cell = cell_grid[y, x]
			write(io, """
				<rect x="$((x-1)*cellWidth)" y=$((y-1)*cellHeight) width="$(cellWidth)" height="$(cellHeight)" class="$(
				cell === GameOfLife.Live ? "live" : "dead"
				)" />
				""")
		end
	end
	
	write(io, "</svg>")
end



"""
Generates a cell that's the updated state for cell_grid(2,2).

"""
function step_cell(cell_grid::AbstractMatrix{Cell})
	# cell_grid is a 1:3,1:3 per Julia 1 indexing. Remove the center cell (2,2)
	centerCellIndex = CartesianIndex(2,2)
	neighbour_indices = filter(
		!=(centerCellIndex),
		CartesianIndices(cell_grid)
	)
	neighbours = view(cell_grid, neighbour_indices)

	# Doesn't call convert(Bool)
	# num_live_neighbours = count(neighbours)
	num_live_neighbours = count(==(Live), neighbours)
	@inbounds if cell_grid[centerCellIndex] == Live
		if num_live_neighbours < 2
			return Dead
		# elseif num_live_neighbours == 2 || num_live_neighbours == 3
		elseif num_live_neighbours < 4
			return Live
		else
			return Dead
		end
	else
		if num_live_neighbours == 3
			return Live
		end
		return Dead
	end
end

"""
	step(grid)

Produce the next iteration of the grid.

Originall based on https://julialang.org/blog/2016/02/iteration/ boxcar3 
Now using CircularArray to handle index overflow, because we're going to do more complicated things
"""
function step_grid(cell_grid::AbstractMatrix{Cell})
	out = similar(cell_grid)

	circular_cell_grid = CircularArray(cell_grid)
	
	range = CartesianIndices(cell_grid)
	it_first = first(range)
	
	range_one = oneunit(it_first)
	
	Threads.@threads for it in range
	#for it in range
		# If we can safely view into the cell_grid, do so, otherwise create a new 3x3 matrix
		it_lower = it-range_one
		it_higher = it+range_one
		local_region = it_lower:it_higher
		
		region = view(circular_cell_grid,local_region)
		
		@inbounds out[it] = step_cell(region)
	end
	
	out
end

export step_grid, Cell

end # module
