#=
Equation: ∂u/∂t = ∂²u/∂x²
Zero Boundary Condition
Crank-Nicolson
=#
using LinearAlgebra
using Plots, LaTeXStrings, Format


Δt = 0.001
t1 = 1
num_steps = Integer(t1 / Δt)
N = 100
Δx = 2 / N
xx = -1.0:Δx:1.0


# initial condition
u0 = zeros(Float64, size(xx))
u0[N÷4:3*N÷4] .= 1.0

A = diagm(
    0 => -2 * ones(Float64, (N - 1,)),
    1 => ones(Float64, (N - 2,)),
    -1 => ones(Float64, (N - 2,))
)

u_old = u0[2:end-1]
ulist = [u_old]
tlist = [0.0]

function step(u)
    LHS = I(N - 1) - 0.5 * Δt / Δx^2 * A
    b = u + 0.5 * Δt / Δx^2 * (A * u)
    return LHS \ b
end

function solve!(ulist, tlist, num_steps)
    u_old = ulist[end]
    t = tlist[end]
    for i in 1:1:num_steps
        u_new = step(u_old)
        u_old = u_new
        t += Δt
        if i % 10 == 0
            push!(ulist, u_new)
            push!(tlist, t)
        end
    end
end

solve!(ulist, tlist, num_steps)


anim = @animate for (u, t) ∈ zip(ulist, tlist)
    plot(xx, [0; u; 0], label="\$u_t = u_{xx}\$", dpi=300)
    xlims!(-1, 1)
    ylims!(-0.1, 1.1)
    xlabel!(L"x")
    ylabel!(L"u")
    title!("\$t=$(format(t, precision=2))\$")
end

gif(anim, "figures/HeatEquation.gif", fps=30)
