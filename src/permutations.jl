"""
    A representation of a specific permutation of a Matrix of an Enum.

    This is probably more abstract than is needed, but might pull it out into a stand alone package in future.
"""
struct GridPermutation{EnumT <: Enum}
    key::BigInt
    dims::Tuple{Int,Int}
    enum::Type{EnumT}
end

function GridPermutation(grid::AbstractMatrix{CellT}) where {CellT <: Enum}
    cell_values = instances(CellT)

    keys = map(
        cell -> findfirst(==(cell), cell_values) - 1,
        vec(parent(grid))
    )

    grid_key = BigInt(0)
    for (index, key) = Iterators.enumerate(keys)
        grid_key += BigInt(key) * (BigInt(length(cell_values)) ^ (BigInt(index-1)))
    end

    GridPermutation(grid_key, size(grid), CellT)
end

import Base: convert, collect

function Base.convert(t::Type{MatrixT}, perm::GridPermutation{CellT}) where {CellT <: Enum, MatrixT <: AbstractMatrix{CellT}}
    cell_values = instances(CellT)
    num_values = length(cell_values)
    
    num_cells = prod(perm.dims)

    keys = digits(perm.key, base=num_values, pad=num_cells)

    grid_backing = map(k -> cell_values[k+1], keys)
    grid_backing = reshape(grid_backing, perm.dims)

    t(grid_backing)
end

"""
    Convert a GridPermutation into a Matrix of the appropriate enum type.
"""
Base.collect(perm::GridPermutation) = convert(Matrix{perm.enum}, perm)

"""
    Gives the maximum key possible for a grid of dims size containing cellT enum.
"""
function max_permutation(dims::Tuple{Int,Int}, cellT::Type{EnumT}) where {EnumT <: Enum}
    cell_values = instances(cellT)
    num_values = length(cell_values)
    num_cells = dims[1]*dims[2]
    return BigInt(num_values) ^ BigInt(num_cells)
end
max_permutation(grid::AbstractMatrix) = max_permutation(size(grid), eltype(grid))

random_permutation(dims::Tuple{Int,Int}, cellT::Type{EnumT}) where {EnumT <: Enum} = GridPermutation(
    rand(1:max_permutation(dims, cellT)),
    dims,
    cellT
)
random_permutation(grid::AbstractMatrix) = random_permutation(size(grid), eltype(grid))

# Convenience pseudo-constructor for Grid
Grid(perm::GridPermutation) = convert(Grid{perm.enum}, perm)
