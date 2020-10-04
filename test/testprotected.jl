
include("./imports.jl")

@testset ExtendedTestSet "Protected" begin
    mutable struct C
        x::Int
        y::Int
    end

    mutable struct B
        c::C
    end

    mutable struct A
        b1::B
        b2::B
    end


    makea() = A(
        B(C(1, 2)),
        B(C(3, 4)),
    )

    @testset ExtendedTestSet "fully protected" begin
        a_p = protect(makea(), NamedTuple())  # protect all child structs of `a`
        @test_throws ProtectedException a_p.b1 isa Protected
    end

    @testset ExtendedTestSet "read direct child" begin
        a_p2 = protect(makea(), (b1=Read(),))  # protect everything but `b1`
        @test_nowarn a_p2.b1
        @test_throws ProtectedException a_p2.b1 = C(5, 5)
    end

    @testset ExtendedTestSet "write direct child" begin
        a_p2 = protect(makea(), (b1=Write(),))  # protect everything but `b1`
        @test_nowarn a_p2.b1
        @test_nowarn a_p2.b1 = B(C(5, 5))
    end
    @testset ExtendedTestSet "write nested" begin
        a_p3 = protect(makea(), (b1 = (c = (x=Write(),),),))  # allow mutating only a.b1.c.x
        @test_nowarn a_p3.b1.c.x = 2
        @test_nowarn a_p3.b1.c.x 
        @test a_p3.b1.c isa Protected
        @test_throws ProtectedException a_p3.b1.c.y = 2
    end
end
