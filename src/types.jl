using CircularArrays

export Cell, Grid, Live, Dead

@enum Cell::UInt8 Dead=false Live=true

import Base: convert, ==

Base.convert(::Type{Bool}, cell::Cell) = cell == Live
"""
    Convert from Real types.
    Negative => Live
    0 => Dead
    Positive => Dead
"""
Base.convert(::Type{Cell}, f::Real) = ifelse(signbit(f), Live, Dead)
"""
    Convert to Real type (could be Number? Not useful as we don't convert back).
    Live => Negative 1
    Dead => Positive 1
"""
Base.convert(::Type{T}, cell::Cell) where { T <: Real} = cell == Live ? -one(T) : one(T)

# const Grid = CircularArray{Cell, 2}
# const Grid{CellT} = CircularArray{CellT, 2}
abstract type Grid{CellT} <: AbstractMatrix{CellT} end

"""
    Most specific - this actually constructs the CircularArray.
    All other overrides should end up here.
"""
Grid{CellT}(backing::AbstractMatrix{CellT}) where {CellT} = CircularArray{CellT, 2}(backing)
"""
    For constructing a specific type of Grid from a different cell type.
"""
Grid{CellT}(backing::AbstractMatrix) where { CellT } = Grid{CellT}(convert(Matrix{CellT}, backing))

"""
    For auto detecting the cell type from the backing matrix.
"""
Grid(backing::AbstractMatrix) = Grid{eltype(backing)}(backing)

"""
    Provided for completeness.
"""
==(c::Cell, f::Real) = c == convert(Cell, f)
"""
    Used if step_grid is called on a Grid{<:Real}
"""
==(f::Real, c::Cell) = c == convert(Cell, f)
