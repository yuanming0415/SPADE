import math
from pprint import pprint
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize 
import xformers
TOKENSCON = 77
TOKENS = 75
import json
import numpy as np
import random
import os
torch.set_printoptions(threshold=np.inf)
def generate_position_encoding(height, width, target_area=None, rsp=None, rep=None, csp=None, cep=None):
    #print(rsp,rep,csp,cep)
    #print(height,width)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    position_encoding = torch.full((height, width), 1, device=device)
    rspf=rsp*height
    repf=rep*height
    cspf=csp*width
    cepf=cep*width
    rspf=int(rspf)
    repf=int(repf)
    cspf=int(cspf)
    cepf=int(cepf)
    if target_area == 'base':
        position_encoding[rspf:repf, cspf:cepf] = 1
    else:
        position_encoding[rspf:repf, cspf:cepf] = 2
    return position_encoding
def _memory_efficient_attention_xformers(module, query, key, value, region_mask=None):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value,attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states
def main_forward_diffusers(module,hidden_states,encoder_hidden_states,divide,userpp = False,tokens=[],width = 100,height = 100,step = 0, isxl = False, inhr = None, target_area=None, rs=None, re=None, cs=None, ce=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    position_encoding = generate_position_encoding(height, width, target_area, rsp = rs, rep = re, csp = cs, cep = ce).to(device)
    position_encoding_flat = position_encoding.view(-1) 
    tokens = hidden_states.size(1)
    assert position_encoding_flat.size(0) == tokens
    position_encoding_expanded = position_encoding_flat.unsqueeze(0).expand(hidden_states.size(0), -1).unsqueeze(-1)
    context = encoder_hidden_states
    key = module.to_k(context)
    position_encoding_expanded = position_encoding_expanded.to(key.dtype)
    query = module.to_q(hidden_states)
    query = query * position_encoding_expanded
    #print("query",query.shape)
    value = module.to_v(context)
    query = module.head_to_batch_dim(query)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)
    hidden_states=_memory_efficient_attention_xformers(module, query, key, value)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = module.to_out[0](hidden_states)
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states 
def hook_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "Attention":
            module.forward = hook_forward(self, module)           
def hook_forward(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x= hidden_states
        #print("-------------------------------------")
        #print("x",x.shape)
        context= encoder_hidden_states
        #print("context",context.shape)
        height =self.h
        width =self.w
        if (self.split_ratio[0].start == 0.5 or self.split_ratio[0].end == 0.5 or self.split_ratio[0].cols[0].start == 0.5 or self.split_ratio[0].cols[0].end == 0.5):
            pass
        elif self.split_ratio[0].cols[0].start == 0 and self.split_ratio[0].cols[0].end == 1:
            height = height * 0.5
        elif self.split_ratio[0].start == 0 and self.split_ratio[0].end == 1:
            width = width * 0.5
        x_t = x.size()[1]
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        ha, wa = x_t % latent_h, x_t % latent_w
        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            print("警示1")
            latent_h = int(x_t / latent_w)
        else:
            print("警示2")
        contexts = context.clone()
        def matsepcalc(x,contexts,pn,divide):
            h_states = []
            x_t = x.size()[1]
            (latent_h,latent_w) = split_dims(x_t, height, width, self)
            latent_out = latent_w
            latent_in = latent_h
            tll = self.pt
            i = 0
            outb = None
            if self.usebase:
                #print("base")
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    print("有大于0的时候吗？")
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                i = i + 1
                #print(x.shape[1])
                if x.shape[1] == 4096:
                    out = main_forward_diffusers(module, x, context, divide,userpp =True, width =64, height =64, isxl = self.isxl, target_area = "base", rs = 0, re = 1, cs = 0, ce = 1)
                elif x.shape[1] == 1024:
                    out = main_forward_diffusers(module, x, context, divide,userpp =True, width =32, height =32, isxl = self.isxl, target_area = "base", rs = 0, re = 1, cs = 0, ce = 1)
                else :
                    #print("base第三维度",latent_w,latent_h)
                    out = main_forward_diffusers(module, x, context, divide,userpp =True, width = latent_w, height =latent_h, isxl = self.isxl, target_area = "base", rs = 0, re = 1, cs = 0, ce = 1)
                outb = out.clone()
                #print("outb",outb.shape)
                outb = outb.reshape(outb.size()[0], latent_h, latent_w, outb.size()[2]) 
            sumout = 0
            for drow in self.split_ratio:
                #print("drow.start",drow.start)
                hstart = drow.start
                #print("drow.hend",drow.end)
                hend = drow.end
                #print(hstart,hend)
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    #print("dcell.start",dcell.start)
                    wstart=dcell.start
                    #print("dcell.end",dcell.end)
                    wend=dcell.end
                    context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        print("有大于0的时候吗？")
                        context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    i = i + 1 + dcell.breaks
                    #print(hstart,hend,wstart,wend)
                    #print(x.shape[1])
                    if x.shape[1] == 4096:
                        if hstart == 0.5:
                            hstart = hstart + 0.1
                        elif hend == 0.5:
                            hend = hend - 0.1
                        elif wstart == 0.5:
                            wstart = wstart + 0.1
                        elif wend == 0.5:
                            wend = wend - 0.1
                        out = main_forward_diffusers(module, x, context, divide,userpp =True, width =64, height =64, isxl = self.isxl, target_area = "part", rs = hstart, re = hend, cs = wstart, ce = wend)
                    elif x.shape[1] == 1024:
                        if hstart == 0.5:
                            hstart = hstart + 0.1
                        elif hend == 0.5:
                            hend = hend - 0.1
                        elif wstart == 0.5:
                            wstart = wstart + 0.1
                        elif wend == 0.5:
                            wend = wend - 0.1
                        out = main_forward_diffusers(module, x, context, divide,userpp =True, width =32, height =32, isxl = self.isxl, target_area = "part", rs = hstart, re = hend, cs = wstart, ce = wend)
                    else :
                        #print("第三维度",latent_w,latent_h)
                        out = main_forward_diffusers(module, x, context, divide,userpp =True, width =latent_w, height =latent_h, isxl = self.isxl, target_area = "part", rs = 0, re = 1, cs = 0, ce = 1)
                    #print("outc",out.shape)
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2])
                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in*dcell.end) - int(latent_in*dcell.start)
                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout = sumout + int(latent_out*drow.end) - int(latent_out*drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out
                    #print(drow.start,drow.end,dcell.start,dcell.end)
                    if x.shape[1] == 4096 or x.shape[1] == 1024:
                        if drow.start == 0 and drow.end == 1:
                            if dcell.start == 0:
                                modend = dcell.end - 0.1
                                out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                    int(latent_w*dcell.start) + addin:int(latent_w*modend),:]
                            elif dcell.end == 1:
                                modstart = dcell.start + 0.1
                                out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                    int(latent_w*modstart) + addin:int(latent_w*dcell.end),:]
                        elif dcell.start == 0 and dcell.end == 1:
                            if drow.start == 0:
                                modend = drow.end - 0.1
                                out = out[:,int(latent_h*drow.start) + addout:int(latent_h*modend),
                                    int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                            elif drow.end == 1:
                                modstart = drow.start + 0.1
                                out = out[:,int(latent_h*modstart) + addout:int(latent_h*drow.end),
                                    int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                    else:
                        #print(int(latent_h*drow.start),int(latent_h*drow.end),int(latent_w*dcell.start),int(latent_w*dcell.end))
                        out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                    if self.usebase :
                        outb_t = outb[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                        int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:].clone()
                        if x.shape[1] == 4096 or x.shape[1] == 1024:
                            if drow.start == 0 and drow.end == 1:
                                if dcell.start == 0:
                                    out_full = torch.zeros_like(outb_t)
                                    if (out_full.size()[2] == 32):
                                        out_full[:, :, :25, :] = out
                                        out_full[:, :, 25:, :] = outb_t[:, :, 25:, :]
                                    elif (out_full.size()[2] == 16):
                                        out_full[:, :, :12, :] = out
                                        out_full[:, :, 12:, :] = outb_t[:, :, 12:, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                                elif dcell.end == 1:
                                    out_full = torch.zeros_like(outb_t)
                                    if (out_full.size()[2] == 32):
                                        n = out_full.size(2)
                                        out_full[:, :, -26:, :] = out
                                        out_full[:, :, :n-26, :] = outb_t[:, :, :n-26, :]
                                    elif (out_full.size()[2] == 16):
                                        n = out_full.size(2)
                                        out_full[:, :, -13:, :] = out
                                        out_full[:, :, :n-13, :] = outb_t[:, :, :n-13, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                            elif dcell.start == 0 and dcell.end == 1:
                                if drow.start == 0:
                                    out_full = torch.zeros_like(outb_t)
                                    if (out_full.size()[1] == 32):
                                        out_full[:, :25, :, :] = out
                                        out_full[:, :, 25:, :] = outb_t[:, :, 25:, :]
                                    elif (out_full.size()[1] == 16):
                                        out_full[:, :12, :, :] = out
                                        out_full[:, :, 12:, :] = outb_t[:, :, 12:, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                                elif drow.end == 1:
                                    out_full = torch.zeros_like(outb_t)
                                    if (out_full.size()[1] == 32):
                                        n = out_full.size(1)
                                        out_full[:, -26:, :, :] = out
                                        out_full[:, :, :n-26, :] = outb_t[:, :, :n-26, :]
                                    elif (out_full.size()[1] == 16):
                                        n = out_full.size(1)
                                        out_full[:, -13:, :, :] = out
                                        out_full[:, :, :n-13, :] = outb_t[:, :, :n-13, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                        else:
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                    v_states.append(out)
                output_x = torch.cat(v_states,dim = 2)
                h_states.append(output_x)
            output_x = torch.cat(h_states,dim = 1) 
            output_x = output_x.reshape(x.size()[0],x.size()[1],x.size()[2])
            #print("output_x",output_x.shape)
            return output_x
        if x.size()[0] == 1 * self.batch_size:
            output_x = matsepcalc(x, contexts, self.pn, 1)
        else:
            if self.isvanilla:
                nx, px = x.chunk(2)
                conn,conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp,conn = contexts.chunk(2)
            opx = matsepcalc(px, conp, True, 2)
            onx = matsepcalc(nx, conn, False, 2)
            if self.isvanilla:
                output_x = torch.cat([onx, opx])
            else:
                output_x = torch.cat([opx, onx]) 
        self.pn = not self.pn
        self.count = 0
        return output_x
    return forward
def split_dims(x_t, height, width, self=None):
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2
    return latent_h, latent_w
def repeat_div(x,y):
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x
