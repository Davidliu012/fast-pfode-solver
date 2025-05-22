from cleanfid import fid
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms

# --------Utility Helper Function--------
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# --------Image Quality Evaluation Function--------
def fid_evaluate(sample_path, gt_path, num_workers):
    """
        Compute the Fréchet inception distance (FID) score between two image folders
    """
    fid_score = fid.compute_fid(sample_path, gt_path, num_workers=num_workers)
    return fid_score


def kid_evaluate(sample_path, gt_path, num_workers):
    """
        Compute the Kernel Inception Distance (KID) score between two image folders
    """
    kid_score = fid.compute_kid(sample_path, gt_path, num_workers=num_workers)
    return kid_score

def IS_evaluate(sample_path, gt_path):
    """
        Compute the Inception Score (IS) between two image folders
    """
    sample_images = load_images_from_folder(sample_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    preds = []
    for img in sample_images:
        img = cv2.resize(img, (75, 75))
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = F.softmax(output, dim=1).cpu().numpy()
            preds.append(pred)

    preds = np.vstack(preds)
    py = np.mean(preds, axis=0)
    scores = []
    for pred in preds:
        kl_div = pred * (np.log(pred + 1e-10) - np.log(py + 1e-10))
        scores.append(np.sum(kl_div))
    is_score = np.exp(np.mean(scores))
    return is_score

def psnr_evaluate(sample_path, gt_path):
    """
        Compute the PSNR score between two image folders
    """
    
    sample_images = load_images_from_folder(sample_path)
    gt_images = load_images_from_folder(gt_path)

    psnr_values = []
    for sample_img, gt_img in zip(sample_images, gt_images):
        sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Calculate PSNR
        psnr_value = cv2.PSNR(sample_img_rgb, gt_img_rgb)
        psnr_values.append(psnr_value)

    return np.mean(psnr_values)

def ssim_evaluate(sample_path, gt_path):
    """
        Compute the SSIM score between two image folders
    """
    
    sample_images = load_images_from_folder(sample_path)
    gt_images = load_images_from_folder(gt_path)

    ssim_values = []
    for sample_img, gt_img in zip(sample_images, gt_images):
        sample_img_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
        gt_img_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_value, _ = ssim(sample_img_gray, gt_img_gray, full=True)
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)


if __name__ == "__main__":
    sampling_results_path = "/home/jingbow/Desktop/PF-ODE/score_sde_pytorch/samples_from_ode/euler-100"
    real_dataset_path = "real_cifar10"
    num_workers = 16

    fid_score = fid_evaluate(sample_path=sampling_results_path, gt_path=real_dataset_path, num_workers=num_workers)
    kid_score = kid_evaluate(sample_path=sampling_results_path, gt_path=real_dataset_path, num_workers=num_workers)
    IS_score = IS_evaluate(sample_path=sampling_results_path, gt_path=real_dataset_path)
    # psnr_score = psnr_evaluate(sample_path=sampling_results_path, gt_path=real_dataset_path)
    # ssim_score = ssim_evaluate(sample_path=sampling_results_path, gt_path=real_dataset_path)

    print(f"FID score between {sampling_results_path} and {real_dataset_path} = {fid_score:.4f}")
    print(f"KID score between {sampling_results_path} and {real_dataset_path} = {kid_score:.4f}")
    print(f"IS score between {sampling_results_path} and {real_dataset_path} = {IS_score:.4f}")
    
    # print("[WARNING] PSNR and SSIM scores are not reliable for randomly generated images.")
    # print(f"PSNR score between {sampling_results_path} and {real_dataset_path} = {psnr_score:.4f}")
    # print(f"SSIM score between {sampling_results_path} and {real_dataset_path} = {ssim_score:.4f}")