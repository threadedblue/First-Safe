using SafeTensors, Flux, FileIO, Images
include("first-hug.jl")
include("first-clip.jl")
include("first-unet.jl")
include("first-vae.jl")
include("first-sched.jl")

# Load safetensors file
const model_path::String = "sd3.5_large.safetensors"

model_data = load_safetensor(model_path)

# Access weights and map them to Flux.jl-compatible structures
unet_weights = model_data["unet"]
vae_weights = model_data["vae"]
text_encoder_weights = model_data["text_encoder"]

# Example scheduler in Julia
function ddim_scheduler(noise, steps)
    # Implement the denoising algorithm
end

function decode_with_vae(latent::Array{Float32}, vae_model::Chain)
    return vae_model(latent)  # VAE decoder reconstructs the pixel image
end

vae_model = create_vae(input_dim, latent_dim, output_dim)

latent = randn(Float32, latent_dims...)  # Initialize latent space
for step in 1:num_steps
    latent = denoise(latent, unet_model, scheduler, text_embeddings)
end

save("output.png", pixel_image)

pixel_image = decode_with_vae(latent, vae_model)

save("output.png", pixel_image)
