using Flux

function scheduler(step::Int, total_steps::Int, schedule_type::Symbol; kwargs...)
    if schedule_type == :linear
        return linear_schedule(step, total_steps)
    elseif schedule_type == :cosine
        return cosine_schedule(step, total_steps)
    elseif schedule_type == :exponential
        return exponential_schedule(step, total_steps, kwargs[:decay_rate])
    else
        throw(ArgumentError("Unknown schedule type: $schedule_type"))
    end
end

function linear_schedule(step::Int, total_steps::Int; start_value::Float64 = 1.0, end_value::Float64 = 0.0)
    return start_value + (end_value - start_value) * (step / total_steps)
end

function cosine_schedule(step::Int, total_steps::Int; start_value::Float64 = 1.0, end_value::Float64 = 0.0)
    cosine_factor = 0.5 * (1 + cos(π * step / total_steps))
    return end_value + (start_value - end_value) * cosine_factor
end

function exponential_schedule(step::Int, total_steps::Int; initial_value::Float64 = 1.0, decay_rate::Float64 = 0.1)
    return initial_value * exp(-decay_rate * step / total_steps)
end

total_steps = 100

for step in 1:total_steps
    value = scheduler(step, total_steps, :linear; start_value=1.0, end_value=0.0)
    println("Step $step: Value = $value")
end
