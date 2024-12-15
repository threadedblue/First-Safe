using Flux
using Flux.Optimise

function encoder(input_dim, latent_dim)
    return Chain(
        Dense(input_dim, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 2 * latent_dim)  # Output both μ and logσ²
    )
end
using Distributions

function sample_latent(params, latent_dim)
    μ = params[1:latent_dim]
    logσ2 = params[latent_dim+1:end]
    σ = exp.(logσ2 ./ 2)
    ε = randn(latent_dim)  # Sample from N(0, I)
    return μ + ε .* σ
end

function decoder(latent_dim, output_dim)
    return Chain(
        Dense(latent_dim, 64, relu),
        Dense(64, 128, relu),
        Dense(128, output_dim, σ -> sigmoid.(σ))  # Output probabilities
    )
end

struct VAE
    encoder
    decoder
    latent_dim
end

function (vae::VAE)(x)
    # Forward pass through encoder
    params = vae.encoder(x)
    z = sample_latent(params, vae.latent_dim)  # Sample latent vector

    # Forward pass through decoder
    return vae.decoder(z), params
end

function create_vae(input_dim, latent_dim, output_dim)
    return VAE(
        encoder(input_dim, latent_dim),
        decoder(latent_dim, output_dim),
        latent_dim
    )
end

function vae_loss(ŷ, y, params, latent_dim)
    # Reconstruction loss (Binary Cross-Entropy)
    recon_loss = -sum(y .* log.(ŷ + 1e-8) + (1 .- y) .* log.(1 .- ŷ + 1e-8))

    # KL Divergence
    μ = params[1:latent_dim]
    logσ2 = params[latent_dim+1:end]
    kl_div = -0.5 * sum(1 .+ logσ2 .- μ.^2 .- exp.(logσ2))

    return recon_loss + kl_div
end

function train_vae(vae, data, epochs, opt, latent_dim)
    for epoch in 1:epochs
        for (x, _) in data  # Assuming data is a DataLoader
            loss, grads = Flux.withgradient(() -> begin
                ŷ, params = vae(x)
                vae_loss(ŷ, x, params, latent_dim)
            end, Flux.params(vae))

            Flux.Optimise.update!(opt, Flux.params(vae), grads)
        end
        println("Epoch $epoch completed.")
    end
end

#6. Example Usage
# Define dimensions
input_dim = 784  # For MNIST, flattened 28x28 images
latent_dim = 20
output_dim = input_dim  # Reconstructed image has same dimensions

# Create VAE
vae = create_vae(input_dim, latent_dim, output_dim)

# Training data (e.g., MNIST)
x_train = rand(Float32, input_dim, 1000)  # Dummy data for demonstration
data_loader = [(x_train, x_train)]  # Use data as both input and target

# Optimizer
opt = ADAM(1e-3)

# Train
train_vae(vae, data_loader, epochs=10, opt=opt, latent_dim=latent_dim)
