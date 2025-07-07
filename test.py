"""Test script for the PADUM (Pixel Adaptive Deep Unfolding Network) model."""
import os
import torch
from basicsr.data.derain_paired_dataset import DerainPairedDataset
from basicsr.models.archs.PADM_arch import PADM

def main():
    """Main function to run the test process."""
    # Configuration parameters
    data_root = 'datasets/test/Rain200L'  # Root directory for test data
    result_dir = 'results/PADUM'         # Directory to save output results
    model_path = 'pretrained_models/PADUM_Rain200L.pth'  # Path to the pretrained model
    
    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize dataset and data loader
    dataset = DerainPairedDataset(data_root=data_root, split='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize model
    model = PADM()
    
    # Load pretrained weights
    if os.path.isfile(model_path):
        print(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"Warning: Pretrained model not found at {model_path}")
        print("Using randomly initialized weights instead")
    
    # Set model to evaluation mode
    model.eval()
    
    # Process each sample in the test dataset
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            lq = data['lq']  # Low-quality (rainy) image
            gt = data['gt']  # Ground truth (clean) image
            filenames = data['lq_path']  # Filenames of the input images
            
            # Forward pass
            output = model(lq)
            
            # Save output images
            for j in range(output.shape[0]):
                filename = os.path.basename(filenames[j])
                result_path = os.path.join(result_dir, filename)
                
                # Convert tensor to PIL image and save
                result_img = transforms.ToPILImage()(output[j].cpu())
                result_img.save(result_path)
                
                print(f"Saved result to {result_path}")

if __name__ == '__main__':
    main()