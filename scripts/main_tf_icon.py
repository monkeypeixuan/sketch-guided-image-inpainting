import sys
sys.path.append('/root/autodl-tmp/users/hpx/TF-ICON/')
import argparse, os
import PIL
import torch
import re
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import cv2
import time
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from transformers import CLIPProcessor, CLIPModel
import csv
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn



def load_img(path, SCALE, pad=False, seg=False, target_size=None):
    if seg:
        # Load the input image and segmentation map
        image = Image.open(path).convert("RGB")
        seg_map = Image.open(seg).convert("1")

        # Get the width and height of the original image
        w, h = image.size

        # Calculate the aspect ratio of the original image
        aspect_ratio = h / w

        # Determine the new dimensions for resizing the image while maintaining aspect ratio
        if aspect_ratio > 1:
            new_w = int(SCALE * 256 / aspect_ratio)
            new_h = int(SCALE * 256)
        else:
            new_w = int(SCALE * 256)
            new_h = int(SCALE * 256 * aspect_ratio)

        # Resize the image and the segmentation map to the new dimensions
        image_resize = image.resize((new_w, new_h))
        segmentation_map_resize = cv2.resize(np.array(seg_map).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad the segmentation map to match the target size
        padded_segmentation_map = np.zeros((target_size[1], target_size[0]))
        start_x = (target_size[1] - segmentation_map_resize.shape[0]) // 2
        start_y = (target_size[0] - segmentation_map_resize.shape[1]) // 2
        padded_segmentation_map[start_x: start_x + segmentation_map_resize.shape[0], start_y: start_y + segmentation_map_resize.shape[1]] = segmentation_map_resize

        # Create a new RGB image with the target size and place the resized image in the center
        padded_image = Image.new("RGB", target_size)
        start_x = (target_size[0] - image_resize.width) // 2
        start_y = (target_size[1] - image_resize.height) // 2
        padded_image.paste(image_resize, (start_x, start_y))

        # Update the variable "image" to contain the final padded image
        image = padded_image
    else:
        image = Image.open(path).convert("RGB")
        w, h = image.size        
        print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        w = h = 512
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    
    if pad or seg:
        return 2. * image - 1., new_w, new_h, padded_segmentation_map
    
    return 2. * image - 1., w, h 


def load_model_and_get_prompt_embedding(model, opt, device, prompts, inv=False):
           
    if inv:
        inv_emb = model.get_learned_conditioning(prompts, inv)
        c = uc = inv_emb
    else:
        inv_emb = None
        
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)
        
    return c, uc, inv_emb
    
    
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=gpu)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


def calculate_clip_score(image1, image2, model, processor):
    """
    计算两张图像之间的CLIP评分
    """
    image1 = processor(images=image1, return_tensors="pt")["pixel_values"]
    image2 = processor(images=image2, return_tensors="pt")["pixel_values"]
    # Get image features using CLIP model (without gradients)
    with torch.no_grad():
        embedding_a = model.get_image_features(image1)
        embedding_b = model.get_image_features(image2)

    # Normalize the embeddings to unit vectors
    embedding_a = embedding_a / embedding_a.norm(p=2, dim=-1, keepdim=True)
    embedding_b = embedding_b / embedding_b.norm(p=2, dim=-1, keepdim=True)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)

    return similarity_score.item()

def calculate_text_clip_score(image, text, model, processor):
    """
    计算图像和文本之间的CLIP评分
    """
    inputs = processor(text=[text], images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 获取图像和文本之间的相似度得分
    clip_score = logits_per_image[0].item()
    return clip_score

def calculate_dino_similarity(image1, image2):
    """
    使用DINO模型计算两张图像的相似度
    """
    # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # 加载DINO的图像处理器和模型
    processor = AutoImageProcessor.from_pretrained('facebook/dino-vits16')
    model = AutoModel.from_pretrained('facebook/dino-vits16').to(device)

    with torch.no_grad():
        # 对第一张图像进行预处理并将其移动到指定设备上
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        # 通过模型获取输出
        outputs1 = model(**inputs1)
        # 提取最后一层的隐藏状态
        image_features1 = outputs1.last_hidden_state

        image_features1 = image_features1.mean(dim=1)
   
    with torch.no_grad():
        # 对第二张图像进行预处理并将其移动到指定设备上
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        # 通过模型获取输出
        outputs2 = model(**inputs2)
        # 提取最后一层的隐藏状态
        image_features2 = outputs2.last_hidden_state
        
        image_features2 = image_features2.mean(dim=1)
    
    cos = nn.CosineSimilarity(dim=0)
    
    sim = cos(image_features1[0], image_features2[0]).item()
    # 将相似度值从[-1, 1]范围映射到[0, 1]范围
    sim = (sim + 1) / 2
    return sim

def evaluate_results(inpainted_image, subject_image, text_prompt):
    """
    评估结果，计算主题身份一致性和文本语义一致性
    """
    # 加载CLIP模型和处理器
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 计算主题身份一致性（CLIP-I）
    clip_i_score = calculate_clip_score(inpainted_image, subject_image, model, processor)
    
    # 计算文本语义一致性（CLIP-T）
    clip_t_score = calculate_text_clip_score(inpainted_image, text_prompt, model, processor)
    
    # 计算DINO图像相似度
    dino_similarity = calculate_dino_similarity(inpainted_image, subject_image)
    
    return clip_i_score, clip_t_score, dino_similarity

csv_file_path="/root/autodl-tmp/users/hpx/TF-ICON/scripts/score_tmp.CSV"
# 写入评估结果到CSV文件
def write_to_csv(index, clip_i_score, clip_t_score,dino_similarity):
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([index, clip_i_score, clip_t_score,dino_similarity])


def get_segmentation_map(seg):
    _, _, _, segmentation_map = load_img(None, 1, seg=seg, target_size=(512, 512))  # 这里假设 target_size 为 (100, 100)，可以根据需要修改
    return segmentation_map

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of a doggy, ultra realistic",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--ref-img",
        type=list,
        nargs="?",
        help="path to the input image"
    )
    
    parser.add_argument(
        "--seg",
        type=str,
        nargs="?",
        help="path to the input image"
    )
        
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )

    parser.add_argument(
        "--dpm_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    
    parser.add_argument(
        "--f",
        type=int,
        default=16,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpt/v2-1_512-ema-pruned.ckpt",
        help="path to checkpoint of model",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="the seed (for reproducible sampling)",
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    parser.add_argument(
        "--root",
        type=str,
        help="",
        default='./inputs/cross_domain'
    ) 
    
    parser.add_argument(
        "--domain",
        type=str,
        help="",
        default='cross'
    ) 
    
    parser.add_argument(
        "--dpm_order",
        type=int,
        help="",
        choices=[1, 2, 3],
        default=2
    ) 
    
    parser.add_argument(
        "--tau_a",
        type=float,
        help="",
        default=0.4
    )
      
    parser.add_argument(
        "--tau_b",
        type=float,
        help="",
        default=0.8
    )
          
    parser.add_argument(
        "--gpu",
        type=str,
        help="",
        default='cuda:0'
    ) 
    
    opt = parser.parse_args()       
    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # The scale used in the paper
    if opt.domain == 'cross':
        opt.scale = 5.0
        file_name = "cross_domain"
    elif opt.domain == 'same':
        opt.scale = 2.5
        file_name = "same_domain"
    else:
        raise ValueError("Invalid domain")
        
    batch_size = opt.n_samples
    sample_path = "/root/autodl-tmp/users/hpx/Evaluation Dataset/output_tmp2"
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, opt.gpu)    
    model = model.to(device)
    sampler = DPMSolverSampler(model)
    
     # 读取CSV文件
    csv_file = "/root/autodl-tmp/users/hpx/Evaluation Dataset/data.CSV"  # 假设opt中新增了csv_file属性来指定CSV文件路径
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            torch.cuda.empty_cache()
            # 获取参数
            bg_id = row['bg_id']
            fg_id = row['fg_id']
            prompt = row['prompt']

            # 假设根据bg_id和fg_id构造文件路径
            opt.init_img = f"/root/autodl-tmp/users/hpx/Evaluation Dataset/background/{bg_id}.jpg"
            opt.ref_img = f"/root/autodl-tmp/users/hpx/Evaluation Dataset/foreground/{fg_id}.jpg"
            opt.mask = f"/root/autodl-tmp/users/hpx/Evaluation Dataset/background/mask_{bg_id}.jpg"  # 这里的路径构造需要根据实际情况调整
            opt.seg = f"/root/autodl-tmp/users/hpx/Evaluation Dataset/foreground/mask_{fg_id}.jpg"  # 这里的路径构造需要根据实际情况调整
            sketch_path = f"/root/autodl-tmp/users/hpx/Evaluation Dataset/sketch/{prompt}.png" 
            
            seed_everything(opt.seed)
            img = cv2.imread(opt.mask, 0)
            img = cv2.resize(img, (512, 512))
            # Threshold the image to create binary image
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            # Find the contours of the white region in the image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the bounding rectangle of the largest contour
            x, y, new_w, new_h = cv2.boundingRect(contours[0])
            # Calculate the center of the rectangle
            center_x = x + new_w / 2
            center_y = y + new_h / 2
            # Calculate the percentage from the top and left
            center_row_from_top = round(center_y / 512, 2)
            center_col_from_left = round(center_x / 512, 2)

            aspect_ratio = new_h / new_w
                
            if aspect_ratio > 1:  
                scale = new_w * aspect_ratio / 256  
                scale = new_h / 256
            else:  
                scale = new_w / 256
                scale = new_h / (aspect_ratio * 256) 
                     
            scale = round(scale, 2)
            
            prompt = f"A professional, ultra-realistic photograph of {prompt}."
                
            # =============================================================================================
        
            assert prompt is not None
            data = [batch_size * [prompt]]
            
            # read background image              
            assert os.path.isfile(opt.init_img)
            init_image, target_width, target_height = load_img(opt.init_img, scale)
            init_image = repeat(init_image.to(device), '1 ... -> b ...', b=batch_size)
            save_image = init_image.clone()
            #subject_image,w,h=load_img(opt.ref_img, scale)
            

            # read foreground image and its segmentation map
            ref_image, width, height, segmentation_map  = load_img(opt.ref_img, scale, seg=opt.seg, target_size=(target_width, target_height))
            ref_image = repeat(ref_image.to(device), '1 ... -> b ...', b=batch_size)
            
            mask_map = Image.open(opt.mask).convert("1")
            mask_map_resize = cv2.resize(np.array(mask_map).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            # 找到 mask 中非零像素的坐标
            nonzero_indices = np.nonzero(mask_map_resize)
            min_y, max_y = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
            min_x, max_x = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
            
            #sketch_path = "/root/autodl-tmp/users/hpx/TF-ICON/inputs/same_domain/a professional photograph of a people and a dog, ultra realistic/dogSketch.png"
            sketch_image, w, h,s  = load_img(sketch_path, scale, seg= opt.seg , target_size=(target_width, target_height))
            sketch_image = repeat(sketch_image.to(device), '1 ... -> b ...', b=batch_size)

            segmentation_map_orig = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
            segmentation_map_save = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 3 ...', b=batch_size)
            segmentation_map = segmentation_map_orig[:, :, ::8, ::8].to(device)
            
            mask_map_orig = repeat(torch.tensor(mask_map_resize)[None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
            mask_map = mask_map_orig[:, :, ::8, ::8].to(device)         

            top_rr = int((0.5*(target_height - height))/target_height * init_image.shape[2])  # xx% from the top
            bottom_rr = int((0.5*(target_height + height))/target_height * init_image.shape[2])  
            left_rr = int((0.5*(target_width - width))/target_width * init_image.shape[3])  # xx% from the left
            right_rr = int((0.5*(target_width + width))/target_width * init_image.shape[3]) 

            center_row_rm = int(center_row_from_top * target_height)
            center_col_rm = int(center_col_from_left * target_width)

            step_height2, remainder = divmod(height, 2)
            step_height1 = step_height2 + remainder
            step_width2, remainder = divmod(width, 2)
            step_width1 = step_width2 + remainder
            
            # compositing in pixel space for same-domain composition
            save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] \
                    = save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2].clone() \
                    * (1 - segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]) \
                    + ref_image[:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                    * segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]
            

            #image_path = '/root/autodl-tmp/users/hpx/TF-ICON/inputs/same_domain/a professional photograph of a dog, ultra realistic/cp_4535_507322.jpg'
            #save_image,w,h = load_img(image_path,scale)
            #save_image = repeat(save_image.to(device), '1 ... -> b ...', b=batch_size)

            # save the mask and the pixel space composited image
            save_mask = torch.zeros_like(init_image) 
            save_mask[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] = 1

            # image = Image.fromarray(((save_mask) * 255)[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy())
            # image.save('./outputs/mask_bg_fg.jpg')
            image = Image.fromarray(((save_image/torch.max(save_image.max(), abs(save_image.min())) + 1) * 127.5)[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy())
            image.save('./outputs/cp_bg_fg.jpg')
            #subject_image = image

            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            
            # image composition
            with torch.no_grad():
                with precision_scope("cuda"):
                    for prompts in data:
                        print(prompts)
                        c, uc, inv_emb = load_model_and_get_prompt_embedding(model, opt, device, prompts, inv=True)
                        
                        if opt.domain == 'same': # same domain

                            init_image = save_image 
                        
                        T1 = time.time()
                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  
                        
        
                        # ref's location in ref image in the latent space
                        top_rr = int((0.5*(target_height - height))/target_height * init_latent.shape[2])  
                        bottom_rr = int((0.5*(target_height + height))/target_height * init_latent.shape[2])  
                        left_rr = int((0.5*(target_width - width))/target_width * init_latent.shape[3])  
                        right_rr = int((0.5*(target_width + width))/target_width * init_latent.shape[3]) 
                                                
                        new_height = bottom_rr - top_rr
                        new_width = right_rr - left_rr
                        
                        step_height2, remainder = divmod(new_height, 2)
                        step_height1 = step_height2 + remainder
                        step_width2, remainder = divmod(new_width, 2)
                        step_width1 = step_width2 + remainder
                        
                        center_row_rm = int(center_row_from_top * init_latent.shape[2])
                        center_col_rm = int(center_col_from_left * init_latent.shape[3])
                        
                        param = [max(0, int(center_row_rm - step_height1)), 
                                min(init_latent.shape[2] - 1, int(center_row_rm + step_height2)),
                                max(0, int(center_col_rm - step_width1)), 
                                min(init_latent.shape[3] - 1, int(center_col_rm + step_width2))]
                        
                        sketch_latent = model.get_first_stage_encoding(model.encode_first_stage(sketch_image))
                    
                        shape = [init_latent.shape[1], init_latent.shape[2], init_latent.shape[3]]
                        z_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                inv_emb=inv_emb,
                                                unconditional_conditioning=uc,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                eta=opt.ddim_eta,
                                                order=opt.dpm_order,
                                                x_T=init_latent,
                                                width=width,
                                                height=height,
                                                DPMencode=True,
                                                )
                        
                        z_ref_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                    inv_emb=inv_emb,
                                                    unconditional_conditioning=uc,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    eta=opt.ddim_eta,
                                                    order=opt.dpm_order,
                                                    x_T=sketch_latent,
                                                    DPMencode=True,
                                                    width=width,
                                                    height=height,
                                                    ref=True,
                                                    )
                        
                        samples_orig = z_enc.clone()

                        # inpainting in XOR region of M_seg and M_mask
                        z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                            = z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                            * segmentation_map[:, :,  top_rr:bottom_rr, left_rr:right_rr] \
                            + torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) \
                            * (1 - segmentation_map[:, :,  top_rr:bottom_rr, left_rr:right_rr])

    
                        samples_for_cross = samples_orig.clone()
                        samples_ref = z_ref_enc.clone()
                        samples = z_enc.clone()

                        # noise composition
                        if opt.domain == 'cross': 
                            samples[:, :, param[0]:param[1], param[2]:param[3]] = torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) 
                            # apply the segmentation mask on the noise
                            samples[:, :, param[0]:param[1], param[2]:param[3]] \
                                    = samples[:, :, param[0]:param[1], param[2]:param[3]].clone() \
                                    * (1 - segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]) \
                                    + z_ref_enc[:, :, top_rr: bottom_rr, left_rr: right_rr].clone() \
                                    * segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]
                        
                        mask = torch.zeros_like(z_enc, device=device)
                        mask[:, :, param[0]:param[1], param[2]:param[3]] = 1
                                            
                        samples, _ = sampler.sample(steps=opt.dpm_steps,
                                                    inv_emb=inv_emb,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    order=opt.dpm_order,
                                                    x_T=[samples_orig, samples.clone(), samples_for_cross, samples_ref, samples, init_latent],
                                                    width=width,
                                                    height=height,
                                                    segmentation_map=segmentation_map,
                                                    param=param,
                                                    mask=mask,
                                                    target_height=target_height, 
                                                    target_width=target_width,
                                                    center_row_rm=center_row_from_top,
                                                    center_col_rm=center_col_from_left,
                                                    tau_a=opt.tau_a,
                                                    tau_b=opt.tau_b,
                                                    sketch=sketch_image,
                                                    )
                            
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        
                        T2 = time.time()
                        print('Running Time: %s s' % ((T2 - T1)))
                        
                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            local_image_np = x_sample[min_y:max_y+1, min_x:max_x+1, :]
                            """
                            subject_image = Image.open(opt.ref_img)
                            mask = mask.resize((512, 512), resample=Image.Resampling.BILINEAR)
                            mask = mask.convert("L")  # 转为灰度图
                            # 使用mask进行区域提取（将非mask区域置为黑色）
                            mask_np = np.array(mask)
                            masked_region = np.zeros_like(x_sample)
                            masked_region[mask_np > 0] = x_sample[mask_np > 0]
                            # 将提取的区域转换为PIL图像格式
                            masked_region = (masked_region * 255).astype(np.uint8)
                            masked_region_image = Image.fromarray(masked_region)
                
                            # 调整图像大小为目标尺寸（默认为512x512
                            masked_region_image = masked_region_image.resize((512, 512), resample=Image.Resampling.BILINEAR)
                            subject_image = Image.open(opt.ref_img)
                            subject_image = subject_image.resize((512, 512), resample=Image.Resampling.BILINEAR)
                            """
                            
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(sample_path, f"{base_count:05}_{prompts[0]}.png"))
                            
                            # 进行评估
                            #先截取生成的部分local_image
                            local_image=Image.fromarray(local_image_np.astype(np.uint8))
                            local_image = local_image.resize((512, 512), resample=Image.Resampling.BILINEAR)
                            subject_image = Image.open(opt.ref_img)
                            subject_image = subject_image.resize((512, 512), resample=Image.Resampling.BILINEAR)
                            
                            clip_i_score, clip_t_score, dino_similarity = evaluate_results(local_image, subject_image, prompts[0])
                            print(f"主题身份一致性（CLIP-I）评分: {clip_i_score}")
                            print(f"文本语义一致性（CLIP-T）评分: {clip_t_score}")
                            print(f"DINO评分: {dino_similarity}")
                            # 写入评估结果到CSV文件
                            write_to_csv(base_count, clip_i_score, clip_t_score,dino_similarity)
                            base_count += 1

            del x_samples, samples, z_enc, z_ref_enc, samples_orig, samples_for_cross, samples_ref, mask, x_sample, img, c, uc, inv_emb
            del param, segmentation_map, top_rr, bottom_rr, left_rr, right_rr, target_height, target_width, center_row_rm, center_col_rm
            del init_image, init_latent, save_image, ref_image, sketch_latent, prompt, prompts, data, binary, contours
        
        print(f"Your samples are ready and waiting for you here: \n{sample_path} \nEnjoy.")

if __name__ == "__main__":
    main()

