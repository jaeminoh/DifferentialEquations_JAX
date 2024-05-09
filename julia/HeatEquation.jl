#=
Equation: ∂u/∂t = ∂²u/∂x²
Zero Boundary Condition
Crank-Nicolson
=#
using LinearAlgebra
using Plots
using Serialization


Δt = 0.001
t1 = 3
num_steps = Integer(t1 / Δt)
N = 100
Δx = 2 / N
xx = Array(-1:Δx:1)

# initial condition
u0 = zeros(Float64, size(xx))
u0[N÷4:3*N÷4] .= 1.0

A = diagm(
    0 => -2 * ones(Float64, (N - 1,)),
    1 => ones(Float64, (N - 2,)),
    -1 => ones(Float64, (N - 2,))
)

u_old = u0[2:end-1]
ulist = [u0]

function step(u)
    LHS = I(N - 1) - 0.5 * Δt / Δx^2 * A
    b = u + 0.5 * Δt / Δx^2 * (A * u)
    return LHS \ b
end

for i in 1:1:num_steps
    u_new = step(u_old)
    global u_old = u_new
    if i % 10 == 0
        push!(ulist, u_new)
    end
end


anim = @animate for u ∈ ulist
    fig = plot(legend=false)
    plot!(fig, u, ylim=(-0.1, 1.1))
end

gif(anim, "HeatEquation.gif", fps=60)
