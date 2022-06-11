using Base: show

"""
	Handle displaying the cell grid in a HTML context. Does this by producing a SVG.
	Originally I did this as HTML, but the layout kept crashing my browser at higher cell sizes.
"""
function Base.show(io::IO, mime::MIME"text/html", cell_grid::AbstractMatrix{Cell})
    # AbstractMatrix{Cell} provides all the bounds checking required

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
	
	# CartesianIndices are lazy, so this is efficient
	for i in CartesianIndices(cell_grid)
		@inbounds cell = cell_grid[i]
		# Iteration (and thus destructuring) of CartesianIndex is specifically not supported
		y, x = Tuple(i)
		write(io, """
			<rect x="$((x-1)*cellWidth)" y=$((y-1)*cellHeight) width="$(cellWidth)" height="$(cellHeight)" class="$(
			cell === Live ? "live" : "dead"
			)" />
			""")
	end
	
	write(io, "</svg>")
end

using ColorTypes
using FixedPointNumbers
using PNGFiles

"""
    TODO: Can this be more generic? Can we take any kind of image MIME, then use
    ImageIO/FileIO to do the translation from pixels to file?

    TODO: Use https://github.com/JuliaPackaging/Requires.jl to make the above dependencies
    optional.
"""
function Base.show(io::IO, mime::MIME"image/png", grid::AbstractMatrix{Cell})
    pixels = map(grid) do cell
        if cell == Live
            RGB{N0f8}(0,0,0)
        else
            RGB{N0f8}(1,1,1)
        end
    end

    PNGFiles.save(io, pixels)
end