"""SAMPLING ONLY."""
import torch
from PIL import Image
import ptp_scripts.ptp_scripts as ptp
import ptp_scripts.ptp_utils as ptp_utils
# from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from scripts.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from tqdm import tqdm
import cv2
import numpy as np

MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}


class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               steps,
               batch_size,
               shape,
               conditioning=None,
               inv_emb=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               t_start=None,
               t_end=None,
               DPMencode=False,
               order=3,
               width=None,
               height=None,
               ref=False,
               top=None, 
               left=None, 
               bottom=None, 
               right=None,
               segmentation_map=None,
               param=None,
               target_height=None, 
               target_width=None,
               center_row_rm=None,
               center_col_rm=None,
               tau_a=0.8,   #原来为0.4
               tau_b=0.8,
               sketch=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {steps}')

        device = self.model.betas.device
        if x_T is None:
            x = torch.randn(size, device=device)
        else:
            x = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
     
        if DPMencode:
            # x_T is not a list
            model_fn = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=None, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )

            dpm_solver = DPM_Solver(model_fn, ns)
            data, _ = self.low_order_sample(x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=DPMencode)
            
            for step in range(order, steps + 1):
                data = dpm_solver.sample_one_step(data, step, steps, order=order, DPMencode=DPMencode)   
                     
            return data['x'].to(device), None
        else:
            # x_T is a list
            model_fn_decode = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )
            model_fn_gen = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale,
            )
            
            orig_controller = ptp.AttentionStore()
            ref_controller = ptp.AttentionStore()
            cross_controller = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()
            Inject_controller = ptp.AttentionStore()
            
            dpm_solver_decode = DPM_Solver(model_fn_decode, ns)
            dpm_solver_gen = DPM_Solver(model_fn_gen, ns)
            
            #----------------------进行初始采样--------------------------
            
            # decoded background
            ptp_utils.register_attention_control(self.model, orig_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                 width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone())
            orig, orig_controller = self.low_order_sample(x[0], dpm_solver_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode, controller=orig_controller)            
            
            # decoded reference(变为ref_attention,并且输入Edge_mask)
            
            #new:生成边缘掩码
            if sketch.dim() == 4:
                sketch = sketch[0, 0]  #将sketch变为二维(512,512)           
            sketch_np = ((sketch.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            edges = cv2.Canny(sketch_np, 100, 200)
            Edge_mask = torch.from_numpy(edges).float().to(device) / 255.0 #edge_mask也是二维向量(512,512)
            
            ptp_utils.register_ref_attention_control(self.model, ref_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                 width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(),edge_mask=Edge_mask)
            ref, ref_controller = self.low_order_sample(x[3], dpm_solver_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode, controller=ref_controller)
            
            # decode for cross-attention
            ptp_utils.register_attention_control(self.model, cross_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                 width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(), pseudo_cross=True)
            cross, cross_controller = self.low_order_sample(x[2], dpm_solver_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode,
                                                                   controller=cross_controller, ref_init=ref['x'].clone())
            
            #----------------------进行初始生成--------------------------
            
            # generation
            Inject_controller = [orig_controller, ref_controller, cross_controller]
            ptp_utils.register_attention_control(self.model, gen_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                 width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(), inject_bg=True)
            gen, _ = self.low_order_sample(x[4], dpm_solver_gen, steps, order, t_start, t_end, device, 
                                           DPMencode=DPMencode, controller=Inject_controller, inject=True)

            for i in range(len(orig['model_prev_list'])):
                blended = orig['model_prev_list'][i].clone() 
                blended[:, :, param[0] : param[1], param[2] : param[3]] \
                    = gen['model_prev_list'][i][:, :, param[0] : param[1], param[2] : param[3]].clone()
                gen['model_prev_list'][i] = blended.clone()
            
            del orig_controller, ref_controller, cross_controller, gen_controller, Inject_controller
            
            #----------------------进行迭代生成--------------------------
                        
            orig_controller = ptp.AttentionStore()
            ref_controller = ptp.AttentionStore()
            cross_controller = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()
            
                            
            for step in range(order, steps + 1):
                # decoded background
                ptp_utils.register_attention_control(self.model, orig_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                     width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone())
                orig = dpm_solver_decode.sample_one_step(orig, step, steps, order=order, DPMencode=DPMencode)
                
                # decode for cross-attention
                ptp_utils.register_attention_control(self.model, cross_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                     width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(), pseudo_cross=True)
                cross['x'] = orig['x']
                cross = dpm_solver_decode.sample_one_step(cross, step, steps, order=order, DPMencode=DPMencode, ref_init=ref['x'].clone())
                                
                if step < int(tau_a*(steps) + 1 - order):
                    inject = True
                    # decoded reference
                    
                    """
                    #new:生成边缘掩码
                    if sketch.dim() == 4:
                        sketch = sketch[0, 0]  #将sketch变为二维(512,512)           
                    sketch_np = ((sketch.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                    edges = cv2.Canny(sketch_np, 100, 200)
                    Edge_mask = torch.from_numpy(edges).float().to(device) / 255.0 #edge_mask也是二维向量(512,512)
                    """
                    
                    #原代码
                    ptp_utils.register_attention_control(self.model, ref_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                         width, height, top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone())
                    #原代码
                    ref = dpm_solver_decode.sample_one_step(ref, step, steps, order=order, DPMencode=DPMencode)
                    controller = [orig_controller, ref_controller, cross_controller]

                        
                else:
                    inject = False
                    controller = [orig_controller, None, cross_controller]
                
                if step < int(0.4*(steps) + 1 - order):
                    inject_bg = True
                else:
                    inject_bg = False
                    
                # generation
                ptp_utils.register_attention_control(self.model, gen_controller, center_row_rm, center_col_rm, target_height, target_width, width, height, 
                                                     top, left, bottom, right, segmentation_map=segmentation_map[0, 0].clone(), inject_bg=inject_bg)
                gen = dpm_solver_gen.sample_one_step(gen, step, steps, order=order, DPMencode=DPMencode, controller=controller, inject=inject)

                if step < int(tau_b*(steps) + 1 - order): 
                    blended = orig['x'].clone() 
                    blended[:, :, param[0] : param[1], param[2] : param[3]] \
                        = gen['x'][:, :, param[0] : param[1], param[2] : param[3]].clone()
                    gen['x'] = blended.clone()      
     
                
            del orig_controller, ref_controller, cross_controller, gen_controller, controller
            return gen['x'].to(device), None
            
    
    def low_order_sample(self, x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=False, controller=None, inject=False, ref_init=None):
        
        t_0 = 1. / dpm_solver.noise_schedule.total_N if t_end is None else t_end
        t_T = dpm_solver.noise_schedule.T if t_start is None else t_start
        
        total_controller = []
        assert steps >= order
        timesteps = dpm_solver.get_time_steps(skip_type="time_uniform", t_T=t_T, t_0=t_0, N=steps, device=device, DPMencode=DPMencode)
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                   controller=[controller[0][0], controller[1][0], controller[2][0]] if isinstance(controller, list) else controller, 
                                                   inject=inject, ref_init=ref_init)]
            
            total_controller.append(controller)
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = dpm_solver.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                           solver_type='dpmsolver', DPMencode=DPMencode)
                model_prev_list.append(dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                           controller=[controller[0][init_order], controller[1][init_order], controller[2][init_order]] if isinstance(controller, list) else controller,
                                                           inject=inject, ref_init=ref_init))
                total_controller.append(controller)
                t_prev_list.append(vec_t)
        
        return {'x': x, 'model_prev_list': model_prev_list, 't_prev_list': t_prev_list, 'timesteps':timesteps}, total_controller
    
    
    def low_order_sample_enhance(self, x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=False, controller=None, inject=False, ref_init=None):
        # 新增权重控制参数
        ref_weight = 1.5  # 可调节范围1.0-2.5
        layer_weights = {
            'down': 2.0,   # 浅层结构控制增强
            'mid': 1.5,    # 中层特征平衡
            'up': 1.2      # 深层细节保留
        }

        t_0 = 1. / dpm_solver.noise_schedule.total_N if t_end is None else t_end
        t_T = dpm_solver.noise_schedule.T if t_start is None else t_start

        total_controller = []
        assert steps >= order
        timesteps = dpm_solver.get_time_steps(skip_type="time_uniform", t_T=t_T, t_0=t_0, N=steps, device=device, DPMencode=DPMencode)
        assert timesteps.shape[0] - 1 == steps

        def enhance_ref_controller(ref_ctrl, step_idx):
            """层级感知的控制器增强"""
            enhanced = []
            for layer in ref_ctrl:
                # 动态权重调整
                time_weight = 1.0 - 0.5 * (step_idx / steps)  # 随时间衰减
                if 'down' in layer['name']:
                    layer_weight = layer_weights['down'] * time_weight
                elif 'mid' in layer['name']:
                    layer_weight = layer_weights['mid'] * time_weight
                else:
                    layer_weight = layer_weights['up'] * time_weight
                    
                # 键值增强策略
                enhanced_layer = {
                    'k': layer['k'] * (1 + ref_weight * layer_weight),
                    'v': layer['v'] * (1 + ref_weight * layer_weight),
                    'q': layer['q']  # 保持原始query
                }
                enhanced.append(enhanced_layer)
            return enhanced

        with torch.no_grad():
            vec_t = timesteps[0].expand(x.shape[0])
            # 初始步骤的控制器增强
            enhanced_ref = enhance_ref_controller(controller[1][0], 0) if inject else None
            model_prev = dpm_solver.model_fn(
                x, vec_t, DPMencode=DPMencode,
                controller=[controller[0][0], enhanced_ref, controller[2][0]] if inject else controller,
                inject=inject, ref_init=ref_init
            )
            model_prev_list = [model_prev]
            total_controller.append(controller)
            t_prev_list = [vec_t]

            # 多步采样循环
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = dpm_solver.multistep_dpm_solver_update(
                    x, model_prev_list, t_prev_list, vec_t, init_order,
                    solver_type='dpmsolver', DPMencode=DPMencode
                )
                
                # 动态增强当前步骤的ref_controller
                if inject and controller is not None:
                    current_ref = controller[1][init_order]
                    enhanced_ref = enhance_ref_controller(current_ref, init_order)
                    step_controller = [controller[0][init_order], enhanced_ref, controller[2][init_order]]
                else:
                    step_controller = controller

                model_prev = dpm_solver.model_fn(
                    x, vec_t, DPMencode=DPMencode,
                    controller=step_controller,
                    inject=inject, ref_init=ref_init
                )
                model_prev_list.append(model_prev)
                total_controller.append(step_controller)
                t_prev_list.append(vec_t)

        return {'x': x, 'model_prev_list': model_prev_list, 't_prev_list': t_prev_list, 'timesteps':timesteps}, total_controller