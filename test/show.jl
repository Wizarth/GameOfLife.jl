# Test that the show functions run. Doesn't validate their output

using Test

function test_show_html()
    io = IOBuffer()

    show(io, MIME("text/html"), glider)

    html = String(take!(io))
    @test !isempty(html)
end

function test_show_png()
    io = IOBuffer()

    show(io, MIME("image/png"), glider)

    html = String(take!(io))
    @test !isempty(html)
end


test_show_html()
test_show_png()