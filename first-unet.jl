using Flux, Metal

function down_block(in_channels, out_channels)
    return Chain(
        Conv((3, 3), in_channels => out_channels, relu; pad=(1, 1)),
        BatchNorm(out_channels),
        Conv((3, 3), out_channels => out_channels, relu; pad=(1, 1)),
        BatchNorm(out_channels),
        MaxPool((2, 2))
    )
end

function up_block(in_channels, out_channels)
    return Chain(
        ConvTranspose((2, 2), in_channels => out_channels, relu; stride=(2, 2)),
        Conv((3, 3), in_channels => out_channels, relu; pad=(1, 1)),
        BatchNorm(out_channels),
        Conv((3, 3), out_channels => out_channels, relu; pad=(1, 1)),
        BatchNorm(out_channels)
    )
end

function bridge(in_channels, out_channels)
    return Chain(
        Conv((3, 3), in_channels => out_channels, relu; pad=(1, 1)),
        BatchNorm(out_channels),
        Conv((3, 3), out_channels => out_channels, relu; pad=(1, 1)),
        BatchNorm(out_channels)
    )
end

function UNet(input_channels, output_channels)
    # Define the encoder
    encoder = [
        down_block(input_channels, 64),
        down_block(64, 128),
        down_block(128, 256),
        down_block(256, 512)
    ]

    # Define the bridge
    bridge_layer = bridge(512, 1024)

    # Define the decoder
    decoder = [
        up_block(1024, 512),
        up_block(512, 256),
        up_block(256, 128),
        up_block(128, 64)
    ]

    # Final output layer
    final_layer = Conv((1, 1), 64 => output_channels, identity)

    return Chain(
        (x -> begin
            # Encoder forward pass
            skips = []
            for layer in encoder
                x = layer(x)
                push!(skips, x)  # Store skip connections
            end

            # Bridge forward pass
            x = bridge_layer(x)

            # Decoder forward pass with skip connections
            for (layer, skip) in zip(decoder, reverse(skips))
                x = layer(cat(x, skip; dims=3))  # Concatenate skip connections
            end

            # Final output
            final_layer(x)
        end)
    )
end

# Initialize the model
model = UNet(1, 1)  # Example: single-channel input and output

# Create dummy input (e.g., 1 channel, 128x128 image)

input = cu(rand(Float32, 1, 128, 128, 1))  # Batch size 1, GPU tensor

# Forward pass
output = model(input)
println("Output shape: ", size(output))
using Flux.Optimise

# Define loss function
loss_fn(ŷ, y) = Flux.mse(ŷ, y)

# Optimizer
opt = ADAM(1e-4)

# Training loop
for epoch in 1:10
    for (x, y) in data_loader  # Assume `data_loader` is your dataset
        gs = Flux.gradient(() -> loss_fn(model(x), y), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end
    println("Epoch $epoch completed.")
end
