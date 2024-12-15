using Test
using MyPackage  # Replace with your package name

@testset "Unit tests for MyPackage" begin
    @test MyPackage.add(1, 2) == 3
    @test MyPackage.divide(4, 2) == 2
    @test_throws DivideError MyPackage.divide(1, 0)
end
