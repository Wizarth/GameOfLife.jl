using CircularArrays

export Cell, Grid, Live, Dead

@enum Cell::UInt8 Live=true Dead=false

using Base: convert

Base.convert(::Type{Bool}, cell::Cell) = cell == Live

const Grid = CircularArray{Cell, 2}
