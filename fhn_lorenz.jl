using DifferentialEquations, Plots

function fhn!(du, u, p, t)
	du[1] = (u[1]-u[1]*u[1])*(u[1]-p[2]) - u[2]
	du[2] = p[3]*(p[1]*u[1] - u[2])
	return nothing
end

function lor!(du, u, p, t)
	du[1] = p[1]*(u[2]-u[1])
	du[2] = u[1]*(p[2]-u[3]) - u[2]
	du[3] = u[1]*u[2] - p[3]*u[3]
	du .= du .* 0.005
	return nothing
end

function f!(du, u, p, t)
	fhn!(view(du,1:2), view(u,1:2), view(p,1:3), t)
	lor!(view(du,3:5), view(u,3:5), view(p,4:6), t)
	#du[1] = du[1] + 0.05*(u[5]-27.0)
	return nothing
end

u0 = randn(Float64,2)
u1 = randn(Float64,3)

p0 = [0.37; 0.1; 0.01]
p1 = [10.0;28.0; 8/3]

tspan = (0.0, 5000.0)

prob0 = ODEProblem(fhn!, u0, tspan, p0)
prob1 = ODEProblem(lor!, u1, tspan, p1)
prob  = ODEProblem(f!, [u0;u1], tspan, [p0;p1])

sol0  = solve(prob0, Tsit5())
sol1  = solve(prob1, Tsit5())
sol   = solve(prob,  Tsit5())

plot(sol0)
plot!(sol1)
plot!(sol, linestyle=:dash, linewidth=2)

function condition(u,t,integrator)
	u[3]
end
function affect!(integrator)
  integrator.u[1] = integrator.u[1] + rand(Float64)
end
cb = ContinuousCallback(condition,affect!)
prob = remake(prob, tspan=(0.0,10000.0))
sol   = solve(prob, Tsit5(); callback=cb)
plot(sol, vars=(0,1))

