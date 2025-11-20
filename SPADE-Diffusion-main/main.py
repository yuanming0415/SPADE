import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
import random
import xformers

def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value,attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    print('hidden_states',hidden_states.size())
    return hidden_states

class RegionalGenerator:
    def __init__(self,model_id, dtype=torch.float32, device="cuda"):

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')                                             #加载预训练tokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder').eval().to(device, dtype=dtype)        #加载预训练text_encoder(eval)
        
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').eval().to(device, dtype=dtype)                          #加载预训练var(eval)
        self.vae.enable_slicing()
    
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet').eval().to(device, dtype=dtype)                 #加载预训练unet(eval)
        self.unet.set_use_memory_efficient_attention_xformers(True)

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")                                             #加载预训练DDIM

        self.dtype = dtype
        self.device = device

        self.hook_forwards(self.unet) #Rewriting Cross Attention is equivalent to partially rewriting the forward process in UNet

    def encode_prompts(self, prompts):
        
        #text_encoder output the hidden states
        #prompts are based on list

        with torch.no_grad():
            tokens = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors='pt').input_ids.to(self.device)  #把文字转换成token
            embs = self.text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(self.device, dtype = self.dtype)                                           #把token经过encoder转换
        return embs

    #Transform to Pillow
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():    
            images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def __call__(
                self,
                prompts,                            #文本提示 
                negative_prompt,                    #文本负面提示
                batch_size = 4,                     #批量大小
                height:int = 512,                   #图像高度
                width:int = 512,                    #图像宽度
                guidance_scale:float = 7.0,         #引导尺度
                num_inference_steps:int = 50,       #推理步数
                seed=42,                            #随机种子
                base_ratio=0.3,                     #基础比率
                end_steps:float = 1,                #结束步数比例
            ):
        '''
        prompts: base prompt + regional prompt
        '''
        if(seed >= 0):
            self.torch_fix_seed(seed=seed)                                                                          #固定种子
        
        self.base_ratio = base_ratio                                                                                #设置base比率
        

        #[main*b,left*b,right*b,neg*b] chunk multiply base prompt,regional prompt and negative by batch size        #把prompt和negative_prompt复制batch_size次后添加到all_promtps[]中
        all_prompts = []
        
        for prompt in prompts:
            all_prompts.extend([prompt] * batch_size)
        all_prompts.extend([negative_prompt] * batch_size)                                                          #all_prompts:一个len为8的list

        #get text prompt(base prompt, regional prompt, negative prompt)的embedding                                  #promtps --> encoder_prompts --> embedding
        text_embs = self.encode_prompts(all_prompts)                                                                #text_embs：一个[8,51,768]的tensor
                                                                
        #set timestep
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)                                       #设置调度器时间步长
        timesteps = self.scheduler.timesteps                                                                        #获取时间步
        #Intialize the noise [batch_size, 4, height // 8, width // 8] common shape in SD hidden states
        latents = torch.randn(batch_size, 4, height // 8, width // 8).to(self.device, dtype = self.dtype)           #获取初始噪声[2,4,112,80]
        latents = latents * self.scheduler.init_noise_sigma                                                         #调整噪声,其实就是1，没调整

        self.height = height // 8                                                                                   #图像高度
        self.width = width // 8                                                                                     #图像宽度
        self.pixels = self.height * self.width                                                                      #图像像素数

        progress_bar = tqdm(range(num_inference_steps), desc="Total Steps", leave=False)                            #创建一个进度条

        self.double = True                                                                                          #???
        for i,t in enumerate(timesteps):                                                                            #i是索引，t是步数
            latent_model_input = torch.cat([latents] * 2)                                                           #图形缩放成为[4,4,112,80]
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            #attention_double version ending condition
            if i > num_inference_steps * end_steps and self.double:
                print(i)
                cond, _, _, negative = text_embs.chunk(4) #cond, left, right, negative
                text_embs = torch.cat([cond,negative])
                self.double = False
            #predict noise
            with torch.no_grad():                                                                                   #预测噪声
                noise_pred = self.unet(sample = latent_model_input,timestep = t,encoder_hidden_states=text_embs).sample

            #negative CFG
            noise_pred_text, noise_pred_negative= noise_pred.chunk(2)                                               #噪声预测为文本和负面提示[4,4,112,80]分为[2,4,112,80]
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_text - noise_pred_negative)             #结合文本和负面提示的噪声预测


            #Get denoised latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample                                       #更新latents

            progress_bar.update(1)                                                                                  #更新进度条 
        images = self.decode_latents(latents)                                                                       #decode latent获取images,latents为[2,4,112,80]
        
        return images                                                                                               #images已经是2张图的list了

    def hook_forward(self, module):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            context = encoder_hidden_states
            batch_size, sequence_length, _ = hidden_states.shape

            query = module.to_q(hidden_states)

            #copy query
            if self.double:
                #(q_cond, q_uncond) -> (q_cond,q_cond,q_cond,q_uncond)
                query_cond , query_uncond = query.chunk(2)
                query = torch.cat([query_cond, query_cond, query_cond, query_uncond])

            context = context if context is not None else hidden_states
            key = module.to_k(context)
            value = module.to_v(context)

            dim = query.shape[-1]

            query = module.head_to_batch_dim(query)
            key = module.head_to_batch_dim(key)
            value = module.head_to_batch_dim(value)

            # attention, what we cannot get enough of
            # if module._use_memory_efficient_attention_xformers:
            hidden_states = _memory_efficient_attention_xformers(module,query, key, value)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
            # else:
            #     if module._slice_size is None or query.shape[0] // module._slice_size == 1:
            #         hidden_states = module._attention(query, key, value)
            #     else:
            #         hidden_states = module._sliced_attention(query, key, value, sequence_length, dim)

            #[left right] * (1-w) + base * w
            if self.double:
                rate = int((self.pixels // query.shape[1]) ** 0.5) #down sample rate
                
                height = self.height // rate
                width = self.width // rate

                
                cond, left, right, uncond = hidden_states.chunk(4)
                
                #reshape to the image shape
                left = left.reshape(left.shape[0],  height, width, left.shape[2])
                right = right.reshape(right.shape[0],  height, width, right.shape[2])

                #combine 
                double = torch.cat([left[:,:,:width//2,:], right[:,:,width//2:,:]], dim=2)
                double = double.reshape(cond.shape[0], -1,  cond.shape[2])

                #weighted sum
                cond = double * (1 - self.base_ratio) + cond * self.base_ratio

                #cond+uncond
                hidden_states = torch.cat([cond,uncond])

            # linear proj
            hidden_states = module.to_out[0](hidden_states)
            # dropout
            hidden_states = module.to_out[1](hidden_states)

            return hidden_states

        return forward
    
    #rewrite unet forward()
    def hook_forwards(self, root_module: torch.nn.Module):
        for name, module in root_module.named_modules():
            if "attn2" in name and module.__class__.__name__ == "Attention":
                print(f'{name}:{module.__class__.__name__}')
                module.forward = self.hook_forward(module)

    #set random seed
    def torch_fix_seed(self, seed=42):
        # Python random
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

model_id = "../models/anything-v3" # jupyter Version only support SD1.4/1.5/2.0/2.1, if you want to try SDXL, use current full repo
torch.cuda.set_device(0)
pipe = RegionalGenerator(model_id,dtype = torch.float16)

prompt = [
    "masterpiece, best quality, 2girls",
    "masterpiece, best quality, 2girls, black ponytail, red eyes, Military uniform",
    "masterpiece, best quality, 2girls, white twintail , blue eyes, cheongsam",
]
negative_prompt = "worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

images = pipe(prompt,negative_prompt,
              batch_size = 1, #batch size
              num_inference_steps=30, # sampling step
              height = 896,
              width = 640,
              end_steps = 1, # The number of steps to end the attention double version (specified in a ratio of 0-1. If it is 1, attention double version will be applied in all steps, with 0 being the normal generation)
              base_ratio=0.2, # Base ratio, the weight of base prompt, if 0, all are regional prompts, if 1, all are base prompts
              seed = 4396, # random seed
)

import matplotlib.pyplot as plt                     #导入matplot模块
import math                                         #导入math模块
plt.figure(figsize=(20,20))                         #创建一个2000*2000的图像
for i,image in enumerate(images):                   #遍历images，取出索引i和image i在这里是0,1 image是图像
    plt.subplot(math.ceil(len(images)/4),4,i+1)     #把所有图像以每列4个排列，子图索引为i+1
    plt.imshow(np.array(image))
    #plt.axis('off')
plt.savefig('output_image8.png')                    #保存最后的组合图片
