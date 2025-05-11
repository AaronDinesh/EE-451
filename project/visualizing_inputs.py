import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from src.models.YOLOv8Lite import YOLOv8Lite
import torchvision.transforms.functional as T



activations_dict = {}
def get_activation_hook(name):
    def hook(model, input, output):
        activations_dict[name] = output.detach()
    return hook

# Regularization: Total Variation Loss
def tv_loss(img, weight=1e-3):
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = weight * (h_variance + w_variance)
    return loss

def deprocess_image(tensor_img):
    img = tensor_img.clone().detach().squeeze(0) # Remove batch dimension
    img = (img - img.min()) / (img.max() - img.min() + 1e-6) # Normalize to 0-1
    img_np = img.permute(1, 2, 0).cpu().numpy() # C,H,W -> H,W,C
    return np.clip(img_np, 0, 1)


def deprocess_to_intensity(tensor_img_rgb):
    """
    Converts a 3-channel RGB tensor image to a single-channel intensity map (grayscale)
    and normalizes it for colormap visualization.
    """
    img_rgb = tensor_img_rgb.clone().detach().squeeze(0) # [3, H, W]
    if img_rgb.shape[0] != 3:
        raise ValueError("Input tensor must be 3-channel (RGB) to convert to intensity.")

    # Convert to grayscale using standard luminance weights
    # Y = 0.299*R + 0.587*G + 0.114*B
    # These weights are for tensors on the same device as img_rgb
    luminance_weights = torch.tensor([0.299, 0.587, 0.114], device=img_rgb.device).view(3, 1, 1)
    intensity_map_tensor = (img_rgb * luminance_weights).sum(dim=0) # Result is [H, W]

    # Normalize the single intensity map to [0, 1]
    map_min = intensity_map_tensor.min()
    map_max = intensity_map_tensor.max()
    normalized_map = (intensity_map_tensor - map_min) / (map_max - map_min + 1e-6)
    
    return normalized_map.cpu().numpy()



def visualize_filters_for_layer(model, layer_name, target_layer_module, num_filters_to_viz,
                                img_size=(640, 640), num_iterations=150, lr=0.1,
                                l2_reg=1e-6, tv_reg=1e-2, device='cpu', output_dir='output_visualizations',
                                use_colormap=None):

    os.makedirs(output_dir, exist_ok=True)
    layer_output_dir = os.path.join(output_dir, layer_name.replace('.', '_')) # Sanitize name for dir
    os.makedirs(layer_output_dir, exist_ok=True)
    print(f"\n--- Visualizing filters for layer: {layer_name} ---")
    hook_handle = target_layer_module.register_forward_hook(get_activation_hook(layer_name))

    num_actual_filters = target_layer_module.out_channels
    #This allows us to visualize a subset of all the filers in the layer (the first num_filters_to_viz). Set
    #num_filters_to_viz to np.inf to visualize all filters
    filters_to_process = min(num_filters_to_viz, num_actual_filters)
    for filter_idx in tqdm(range(filters_to_process), desc=f"Filters for {layer_name}", position=0, total=filters_to_process, leave=False):
        # Reset random seed for initial image for somewhat deterministic results per filter (optional)
        # torch.manual_seed(filter_idx)
        # np.random.seed(filter_idx)
        
        # Start with random noise image. We start at the center of the 0-1 range so that our values will remain displayable
        optimized_img = (torch.randn(1, 3, img_size[0], img_size[1]) * 0.1 + 0.5).to(device)
        optimized_img.requires_grad_(True)

        #This optimizer will perform the gradient ascent
        optimizer = torch.optim.Adam([optimized_img], lr=lr, weight_decay=l2_reg)

        for i in tqdm(range(num_iterations), desc="Performing gradient ascent", position=1, total=num_iterations, leave=False):
            optimizer.zero_grad()
            
            
            #We do a full forward pass of the model and let the hook we registed before capture the activation of a
            #particular layer.
            _ = model(optimized_img)
            if layer_name not in activations_dict:
                raise RuntimeError(f"Hook failed to capture activations for {layer_name}")
            
            # Get the activation from our hook
            layer_output = activations_dict[layer_name]
            assert not torch.isnan(layer_output).any(), "NaNs in activation!"
            filter_activation = layer_output[0, filter_idx, :, :]
            print(f"Mean activation for filter {filter_idx}: {filter_activation.mean().item():.4f}")

            
            # Compute loss
            loss = torch.mean(filter_activation)
            #We subtract the tv_loss for some regularization
            loss -= tv_loss(optimized_img, weight=tv_reg)
            loss = -loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                optimized_img.data.clamp_(0, 1)

            # Optional: Gaussian blur every few steps to smooth further
            if (i + 1) % 25 == 0:
                optimized_img.data = T.GaussianBlur(kernel_size=3, sigma=(0.5, 0.5))(optimized_img.data)

            if (i + 1) % 10 == 0:
                print(f"Step {i+1}: loss={-loss.item():.4f}, img_std={optimized_img.std().item():.4f}")

        final_pattern_tensor = optimized_img.detach().clone()
        
        
        plt.figure(figsize=(6,5))
        if use_colormap is not None:
            final_pattern_vis = deprocess_to_intensity(final_pattern_tensor)
            plt.imshow(final_pattern_vis, cmap=use_colormap)
            plt.colorbar(label='Normalized Intensity')
            title_suffix = f" ({use_colormap} map)"
        else:
            final_pattern_vis = deprocess_image(final_pattern_tensor)
            plt.imshow(final_pattern_vis)
            title_suffix = " (RGB)"
        
        plt.title(f"Layer: {layer_name}\nFilter: {filter_idx}{title_suffix}", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        safe_layer_name = layer_name.replace('.', '_')
        colormap_tag = f"_{use_colormap}" if use_colormap else "_rgb"
        plt.savefig(os.path.join(layer_output_dir, f"{safe_layer_name}_filter_{filter_idx}{colormap_tag}.png"))
        plt.close()

    hook_handle.remove() # Clean up the hook
    if layer_name in activations_dict: # Clean up dict entry
        del activations_dict[layer_name]

def main():
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Configuration ---
    IMG_SIZE = (640, 640) 

    # --- Visualization Parameters ---
    MAX_FILTERS_PER_LAYER = 4  # Visualize the first N filters of each Conv2D layer
    NUM_ITERATIONS = 400       # Iterations for optimization
    LEARNING_RATE = 0.05       # Learning rate
    L2_REG_STRENGTH = 1e-3     # L2 regularization (via Adam's weight_decay)
    TV_REG_STRENGTH = 1e-2     # Total Variation regularization
    OUTPUT_VIS_DIR = "output_filter_visualizations"
    COLOR_MAP = 'viridis'

    model = YOLOv8Lite().to(device)
    model.load_state_dict(torch.load("./model_weights/CURRENT_BEST.pt"))
    model.eval()

    conv_layers_to_visualize = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name == "head":
                print(f"Skipping visualization for layer: {name} (prediction head)")
                continue

            conv_layers_to_visualize.append((name, module))
            print(f"Found Conv2d layer: {name} with {module.out_channels} filters")
    
    if not conv_layers_to_visualize:
        print("No Conv2d layers found to visualize (or all were skipped). Check model structure and filters.")
    else:
        print(f"\nStarting filter visualization for {len(conv_layers_to_visualize)} Conv2D layers...")

        for layer_name, layer_module in conv_layers_to_visualize:
            visualize_filters_for_layer(
                model=model,
                layer_name=layer_name,
                target_layer_module=layer_module,
                num_filters_to_viz=MAX_FILTERS_PER_LAYER,
                img_size=IMG_SIZE,
                num_iterations=NUM_ITERATIONS,
                lr=LEARNING_RATE,
                l2_reg=L2_REG_STRENGTH,
                tv_reg=TV_REG_STRENGTH,
                device=device,
                output_dir=OUTPUT_VIS_DIR,
                use_colormap=COLOR_MAP
            )
        print("\nFilter visualization complete.")
        print(f"Results saved in: {os.path.abspath(OUTPUT_VIS_DIR)}")

if __name__ == "__main__":
    main()