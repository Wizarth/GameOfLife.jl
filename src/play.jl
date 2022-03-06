using VideoIO
using CUDA
using ProgressMeter

include("GameOfLife.jl")

grid_size = 4096

cpu_cell_grid = rand([
    GameOfLife.Live,
    GameOfLife.Dead,
    GameOfLife.Dead,
    GameOfLife.Dead,
    GameOfLife.Dead
    ], grid_size, grid_size)

cell_grid = CuArray{GameOfLife.Cell}(undef, (grid_size, grid_size))

copy!(cell_grid, cpu_cell_grid)

function Base.convert(::Type{UInt8}, cell::GameOfLife.Cell)
	if cell == GameOfLife.Live
		zero(UInt8)
	else
		typemax(UInt8)
	end
end

encoder_options = (crf=0, preset="ultrafast", color_range=2)

framerate = 20

frame_count = 1000

# ╔═╡ 7134a77f-3dc2-4bda-9c08-7b577a1da189
open_video_out("video.mp4", Matrix{UInt8}(cell_grid), framerate=framerate, encoder_options=encoder_options) do writer
    global cell_grid
	@showprogress 1 "Computing..." for _ in 1:frame_count
		cell_grid = GameOfLife.step_grid(cell_grid)
		write(writer, Matrix{UInt8}(cell_grid))
	end
end	