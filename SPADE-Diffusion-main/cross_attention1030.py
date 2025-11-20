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
    #print("rs,re,cs,ce",rspf,repf,cspf,cepf)
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
    #print("height,width,rs,re,cs,ce",height,width,rs,re,cs,ce)
    position_encoding = generate_position_encoding(height, width, target_area, rsp = rs, rep = re, csp = cs, cep = ce).to(device)
    #print("-----------------shape:",position_encoding.shape)
    position_encoding_flat = position_encoding.view(-1)
    #print("------------------position_encoding",position_encoding_flat.shape)
    tokens = hidden_states.size(1)
    #print("-------------------hidden_states",hidden_states.shape)
    assert position_encoding_flat.size(0) == tokens
    position_encoding_expanded = position_encoding_flat.unsqueeze(0).expand(hidden_states.size(0), -1).unsqueeze(-1)
    context = encoder_hidden_states
    key = module.to_k(context)
    position_encoding_expanded = position_encoding_expanded.to(key.dtype)
    query = module.to_q(hidden_states)
    #print("zd",position_encoding_expanded)
    query = query * position_encoding_expanded
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
        print("开始==================================")
        #print("x",x.shape)
        context= encoder_hidden_states
        #print("context",context.shape)
        #print("context_begin",context.shape)
        #print(hidden_states.shape)
        height =self.h
        width =self.w
        #print("height",height)
        #print("width",width)
        #print("-------------------------------------------------")
        #print(self.split_ratio[0].start)
        #print(self.split_ratio[0].end)
        #print(self.split_ratio[0].cols[0].start)
        #print(self.split_ratio[0].cols[0].end)
        #print("-------------------------------------------------")
        #print(self.split_ratio_all)
        if (hidden_states.shape[1] == 16384 and hidden_states.shape[2] == 640) or \
            (hidden_states.shape[1] == 4096 and hidden_states.shape[2] == 1280):
            # 满足条件时执行的代码
            #print("------------------------------------整体扩散-----------------------------------------------------")
            pass
        else:
            #print("-------------------------------------区域扩散---------------------------------------------------")
            for i in range(len(self.split_ratio_allcp)):
                if self.split_ratio_allcp[i].now == 1:
                    for j in range(len(self.split_ratio_allcp[i].cols)):
                        if self.split_ratio_allcp[i].cols[j].now == 1:
                            text = self.split_ratio_all.strip()
                            rows = []
                            if ';' in text:
                                groups = [g.strip() for g in text.split(';') if g.strip()]
                                for g in groups:
                                    parts = [p.strip() for p in g.split(',') if p.strip()]
                                    if len(parts) == 1:
                                        rows.append([float(parts[0]), 1.0])
                                    else:
                                        rows.append([float(x) for x in parts])
                            elif ',' in text:
                                parts = [p.strip() for p in text.split(',') if p.strip()]
                                rows.append([1.0] + [float(x) for x in parts])
                            else:
                                rows.append([float(text)])
                            v_weights = [r[0] for r in rows]
                            v_ratio = v_weights[i]
                            row_cols = rows[i][1:]
                            h_ratio = row_cols[j]
                            height = height * v_ratio
                            width = width * h_ratio
                            # 检查该区域的 extend 参数是否被修改
                            ext = getattr(self.split_ratio_allcp[i].cols[j], "extend", 1)

                            # --- 判断逻辑 ---
                            if isinstance(ext, dict):
                                # 如果是 dict，就检测任意一条边是否 != 1
                                modified = any(abs(v - 1.0) > 1e-6 for v in [
                                    ext.get('top', 1.0),
                                    ext.get('bottom', 1.0),
                                    ext.get('left', 1.0),
                                    ext.get('right', 1.0)
                                ])
                            else:
                                # 否则就是单值，只要 != 1 就算修改
                                modified = abs(ext - 1.0) > 1e-6

                            #if modified:
                            #    print(f"[INFO] 区域 [{i},{j}] 的 extend 已被修改: {ext}")
                            #else:
                            #    print(f"[INFO] 区域 [{i},{j}] 的 extend 为默认值 (1.0)")

                            # --- 如果 extend 被修改，则根据伸缩比例调整 height 和 width ---
                            if modified:
                                # 确定每个方向的伸缩比例
                                if isinstance(ext, dict):
                                    top_e = ext.get('top', 1.0)
                                    bottom_e = ext.get('bottom', 1.0)
                                    left_e = ext.get('left', 1.0)
                                    right_e = ext.get('right', 1.0)
                                else:
                                    top_e = bottom_e = left_e = right_e = ext

                                # 平均垂直伸缩比例 = 上下两边平均
                                v_scale = (top_e + bottom_e) / 2.0
                                # 平均水平伸缩比例 = 左右两边平均
                                h_scale = (left_e + right_e) / 2.0

                                # 按比例调整区域高宽
                                height *= v_scale
                                width  *= h_scale

                                #print(f"[UPDATE] 区域 [{i},{j}] 调整尺寸: height × {v_scale:.2f}, width × {h_scale:.2f}")


                            #print(f"第{i}行第{j}列 => 纵向比例: {v_ratio:.6f}，横向比例: {h_ratio:.6f}")
            #print("zheliwanchenglema?")
            #print(self.split_ratio)
            #print(self.split_ratio[1].start)
            #print(self.split_ratio[1].end)
            # 不满足条件时执行的代码
        #    for i in range(len(self.split_ratio)):
        #        #print("--",i)
        #        if self.split_ratio[i].now == 1:
        #            #print(f"找到了{i}\n")
        #            #print(self.split_ratio[i].start,self.split_ratio[i].end)
        #            for j in range(len(self.split_ratio[i].cols)):
        #                #print("++",self.split_ratio[i].cols[j].start)
        #                #print("++",self.split_ratio[i].cols[j].end)
        #                if self.split_ratio[i].cols[j].now == 1:
        #                    print(f"找到了{i}和{j}\n")
        #                    print(self.split_ratio[i].cols[j].start,self.split_ratio[i].cols[j].end)
        #                    print(self.split_ratio_all)
        #                    groups = [g.strip() for g in self.split_ratio_all.split(';') if g.strip() != '']
        #                    print("===============",groups)
        #                    rows = []
        #                    for g in groups:
        #                        parts = [p.strip() for p in g.split(',') if p.strip() != '']
        #                        rows.append([float(x) for x in parts])
        #                    print("===============",rows)
        #                    v_weights = [r[0] for r in rows]
        #                    print("--------------",v_weights)
        #                    v_ratio = v_weights[i]
        #                    row_cols = rows[i][1:]
        #                    h_ratio = row_cols[j]
        #                    print(f"第{i}行第{j}列 => 纵向比例: {v_ratio:.6f}，横向比例: {h_ratio:.6f}")

        #print("11111111",self.split_ratio[0].start,self.split_ratio[0].end,self.split_ratio[0].cols[0].start,self.split_ratio[0].cols[0].end)
        #if (self.split_ratio[0].start == 0.5 or self.split_ratio[0].end == 0.5 or self.split_ratio[0].cols[0].start == 0.5 or self.split_ratio[0].cols[0].end == 0.5):
        #    pass
            #print(self.split_ratio[0].start,self.split_ratio[0].end,self.split_ratio[0].cols[0].start,self.split_ratio[0].cols[0].end)
            #print(height,width)
            #print(self.split_ratio[0].now)
        #elif self.split_ratio[0].cols[0].start == 0 and self.split_ratio[0].cols[0].end == 1:
        #    height = height * 0.5
            #print(height,width)
            #print(self.split_ratio[0].now)
        #elif self.split_ratio[0].start == 0 and self.split_ratio[0].end == 1:
        #    width = width * 0.5
            #print(self.split_ratio[0].now)
            #print(height,width)
        #if 
        x_t = x.size()[1]
        #print("-height",height)
        #print("-width",width)
        #print("-x_t",x_t)
        scale = round(math.sqrt(height * width / x_t))
        #print("scale",scale)
        latent_h = round(height / scale)
        #print("-------------------------------")
        #print("latenth",latent_h)
        latent_w = round(width / scale)
        #print("latentw",latent_w)
        #print("-------------------------------")
        ha, wa = x_t % latent_h, x_t % latent_w
        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            print("警示1")
            latent_h = int(x_t / latent_w)
        else:
            print("警示2")
        contexts = context.clone()
        #print("context",context.shape)
        def matsepcalc(x,contexts,pn,divide):
            h_states = []
            x_t = x.size()[1]
            (latent_h,latent_w) = split_dims(x_t, height, width, self)
            latent_out = latent_w
            latent_in = latent_h
            tll = self.pt
            i = 0
            outb = None
            #print("到这里来了没")
            if self.usebase:
                #print("base")
                context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    print("有大于0的时候吗？")
                    context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                i = i + 1
                #print("base")
                #print(height,width)
                #print(x.shape)
                if (x.shape[1] == 16384 and x.shape[2] == 640):
                    #print("11111111111111111111111111111")
                    out = main_forward_diffusers(module, x, context, divide,userpp =True, width =128, height =128, isxl = self.isxl, target_area = "base", rs = 0, re = 1, cs = 0, ce = 1)
                    #print("22222222222222222222222222222222")
                elif (x.shape[1] == 4096 and x.shape[2] == 1280):
                    out = main_forward_diffusers(module, x, context, divide,userpp =True, width =64, height =64, isxl = self.isxl, target_area = "base", rs = 0, re = 1, cs = 0, ce = 1)
                else :
                    #print("fenkaikuosan ",latent_w,latent_h)
                    out = main_forward_diffusers(module, x, context, divide,userpp =True, width = latent_w, height =latent_h, isxl = self.isxl, target_area = "base", rs = 0, re = 1, cs = 0, ce = 1)
                outb = out.clone()
                #print("outb",outb.shape)
                outb = outb.reshape(outb.size()[0], latent_h, latent_w, outb.size()[2]) 
                #print("先做一个base的整体",outb.shape)
            sumout = 0
            #print(len(self.split_ratio))
            #for i, _ in enumerate(self.split_ratio):
            #    print(i)
            # ===== 区域参数批量修改模块 =====


            # 格式说明：
            # 每个元素是一个修改指令：(row_idx, col_idx, new_start, new_end)
            # 如果 col_idx = None，表示修改整行的 start 值。
            #mods = [
            #    (0, 1, 0.25, 0.75),  # 第0行第1列改为 start=0.1 end=0.3
            #    (1, 0, 0.25, 0.75),  # 第1行第0列改为 start=0.2 end=0.5
            #    (2, None, 0.25, 0.75),  # 第2行的 start 改为0.25
            #]

            #for (row_idx, col_idx, new_start, new_end) in mods:
            #    try:
            #        if not (0 <= row_idx < len(self.split_ratio)):
            #            raise IndexError(f"行索引 {row_idx} 超出范围")
            #        target_row = self.split_ratio[row_idx]

            #        if col_idx is None:
            #            print(f"修改行 {row_idx}: start={target_row.start} -> {new_start}")
            #            target_row.start = new_start
            #        else:
            #            if not (0 <= col_idx < len(target_row.cols)):
            #                raise IndexError(f"列索引 {col_idx} 超出范围（行 {row_idx} 中列数 {len(target_row.cols)}）")
            #            target_col = target_row.cols[col_idx]
            #            print(f"修改行 {row_idx} 列 {col_idx}: start={target_col.start}->{new_start}, end={target_col.end}->{new_end}")
            #            target_col.start = new_start
            #            if new_end is not None:
            #                target_col.end = new_end
            #    except Exception as e:
            #        print(f"修改失败（行{row_idx},列{col_idx}）：{e}")
            # ===== 修改模块结束 =====

            if (x.shape[1] == 16384 and x.shape[2] == 640) or (x.shape[1] == 4096 and x.shape[2] == 1280):
                print("整体扩散我要调整注意力机制")
                mods = [
#                            (0, 1, 0, 0.25),  # 示例：第0行第1列改为 start=0.1 end=0.3
                            (1, None, 0.25, None),  # 示例：第1行整体start改为0.25
                       ]

                for (row_idx, col_idx, new_start, new_end) in mods:
                    try:
                        if not (0 <= row_idx < len(self.split_ratio)):
                            raise IndexError(f"行索引 {row_idx} 超出范围")
                        target_row = self.split_ratio[row_idx]

                        if col_idx is None:
                            print(f"修改行 {row_idx}: start={target_row.start} -> {new_start}")
                            target_row.start = new_start
                        else:
                            if not (0 <= col_idx < len(target_row.cols)):
                                raise IndexError(f"列索引 {col_idx} 超出范围（行 {row_idx} 中列数 {len(target_row.cols)}）")
                            target_col = target_row.cols[col_idx]
                            print(f"修改行 {row_idx} 列 {col_idx}: start={target_col.start}->{new_start}, end={target_col.end}->{new_end}")
                            target_col.start = new_start
                            if new_end is not None:
                                target_col.end = new_end
                    except Exception as e:
                        print(f"修改失败（行{row_idx},列{col_idx}）：{e}")
            else:
                print("区域扩散不调整了")
            print("调整完了")
            print("check",self.split_ratio)
            # 假设 self.split_ratio 是一个包含若干 Row 对象的列表
            for iko, rowko in enumerate(self.split_ratio):
                print(f"Row {iko}: start={rowko.start}, end={rowko.end}, now={rowko.now}")
                pass
                # 遍历当前行的每个 Region
                if hasattr(rowko, "cols"):
                    for jko, regionko in enumerate(rowko.cols):
                        print(
                            f"    Col {jko}: start={regionko.start}, end={regionko.end}, "
                            f"base={regionko.base}, now={regionko.now}"
                        )
            for drow in self.split_ratio:
                print("行一次")
                hstart = drow.start
                hend = drow.end
                print(hstart,hend)
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    print("列一次")
                    wstart=dcell.start
                    wend=dcell.end
                    print(wstart,wend)
                    context = contexts[:,tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON,:]
                    cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        print("有大于0的时候吗？")
                        context = torch.cat([context,contexts[:,-cnet_ext:,:]],dim = 1)
                    i = i + 1 + dcell.breaks
                    if (x.shape[1] == 16384 and x.shape[2] == 640):
                        if hstart == 0.5:
                            hstart = hstart
                        elif hend == 0.5:
                            hend = hend
                        elif wstart == 0.5:
                            wstart = wstart
                        elif wend == 0.5:
                            wend = wend
                        #print("整体扩散注意力机制",hstart,hend,wstart,wend)
                        out = main_forward_diffusers(module, x, context, divide,userpp =True, width =128, height =128, isxl = self.isxl, target_area = "part", rs = hstart, re = hend, cs = wstart, ce = wend)
                    elif (x.shape[1] == 4096 and x.shape[2] == 1280):
                        if hstart == 0.5:
                            hstart = hstart
                        elif hend == 0.5:
                            hend = hend
                        elif wstart == 0.5:
                            wstart = wstart
                        elif wend == 0.5:
                            wend = wend
                        #print("整体扩散注意力机制2",hstart,hend,wstart,wend)
                        out = main_forward_diffusers(module, x, context, divide,userpp =True, width =64, height =64, isxl = self.isxl, target_area = "part", rs = hstart, re = hend, cs = wstart, ce = wend)
                    else :
                        out = main_forward_diffusers(module, x, context, divide,userpp =True, width =latent_w, height =latent_h, isxl = self.isxl, target_area = "part", rs = 0, re = 1, cs = 0, ce = 1)
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2])
                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in*dcell.end) - int(latent_in*dcell.start)
                    #if dcell.end >= 0.999:
                    #    addin = sumin - latent_in
                    #    sumout = sumout + int(latent_out*drow.end) - int(latent_out*drow.start)
                    #    if drow.end >= 0.999:
                    #        addout = sumout - latent_out
                    if (x.shape[1] == 16384 and x.shape[2] == 640) or (x.shape[1] == 4096 and x.shape[2] == 1280):
                        #print("裁剪的时候",drow.start,drow.end,dcell.start,dcell.end)
                        out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                        #print("裁剪整体扩散后",out.shape)
                        #if drow.start == 0 and drow.end == 1:
                        #    if dcell.start == 0:
                        #        modend = dcell.end
                        #        out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                        #            int(latent_w*dcell.start) + addin:int(latent_w*modend),:]
                        #    elif dcell.end == 1:
                        #        modstart = dcell.start
                        #        out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                        #            int(latent_w*modstart) + addin:int(latent_w*dcell.end),:]
                        #elif dcell.start == 0 and dcell.end == 1:
                        #    if drow.start == 0:
                        #        modend = drow.end
                        #        out = out[:,int(latent_h*drow.start) + addout:int(latent_h*modend),
                        #            int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                        #    elif drow.end == 1:
                        #        modstart = drow.start
                        #        out = out[:,int(latent_h*modstart) + addout:int(latent_h*drow.end),
                        #            int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                    else:
                        out = out[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:]
                    if self.usebase :
                        #print("抠use图",drow.start,drow.end,dcell.start,dcell.end)
                        outb_t = outb[:,int(latent_h*drow.start) + addout:int(latent_h*drow.end),
                                        int(latent_w*dcell.start) + addin:int(latent_w*dcell.end),:].clone()
                        if (x.shape[1] == 16384 and x.shape[2] == 640) or (x.shape[1] == 4096 and x.shape[2] == 1280):
                            #print("整体")
                            if drow.start == 0 and drow.end == 1:
                                if dcell.start == 0:
                                    out_full = torch.zeros_like(outb_t)
                                    #print(out_full.shape)
                                    if (out_full.size()[2] == 32):
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, 25:, :] = outb_t[:, :, 25:, :]
                                    elif (out_full.size()[2] == 16):
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, 12:, :] = outb_t[:, :, 12:, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                                elif dcell.end == 1:
                                    out_full = torch.zeros_like(outb_t)
                                    if (out_full.size()[2] == 32):
                                        n = out_full.size(2)
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, :n-26, :] = outb_t[:, :, :n-26, :]
                                    elif (out_full.size()[2] == 16):
                                        n = out_full.size(2)
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, :n-13, :] = outb_t[:, :, :n-13, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                            elif dcell.start == 0 and dcell.end == 1:
                                if drow.start == 0:
                                    out_full = torch.zeros_like(outb_t)
                                    if (out_full.size()[1] == 32):
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, 25:, :] = outb_t[:, :, 25:, :]
                                    elif (out_full.size()[1] == 16):
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, 12:, :] = outb_t[:, :, 12:, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                                elif drow.end == 1:
                                    out_full = torch.zeros_like(outb_t)
                                    if (out_full.size()[1] == 32):
                                        n = out_full.size(1)
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, :n-26, :] = outb_t[:, :, :n-26, :]
                                    elif (out_full.size()[1] == 16):
                                        n = out_full.size(1)
                                        out_full[:, :, :, :] = out
                                        #out_full[:, :, :n-13, :] = outb_t[:, :, :n-13, :]
                                    out_full = out_full * (1 - dcell.base) + outb_t * dcell.base
                                    out = out_full
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                        else:
                            #print("区域")
                            out = out * (1 - dcell.base) + outb_t * dcell.base
                    v_states.append(out)
                    print("out",out.shape)
                output_x = torch.cat(v_states,dim = 2)
                h_states.append(output_x)
                print("output_x",output_x.shape)
                #分区for循环就到这里了
            output_x = torch.cat(h_states,dim = 1)
            print("最终",output_x.shape)
            output_x = output_x.reshape(x.size()[0],x.size()[1],x.size()[2])
            print("---------------------==============================--------------------------------------------------")
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
