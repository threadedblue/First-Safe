using HuggingFaceHub

# Authenticate with Hugging Face if needed
# (Set your token as an environment variable or pass it directly)
ENV["HF_HOME"] = "~/.huggingface"  # Optional, sets where models are stored
ENV["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Download a model by name
repo = "stabilityai/stable-diffusion-3.5-large"
model_path = hf_download(repo, subdir="")

println("Model downloaded to: ", model_path)
