import colorsys  # Polygon regions.
from PIL import Image, ImageChops
from pprint import pprint
import cv2  # Polygon regions.
import numpy as np
import PIL
import torch

SPLROW = ";"
SPLCOL = ","
KEYROW = "ADDROW"
KEYCOL = "ADDCOL"
KEYBASE = "ADDBASE"
KEYCOMM = "ADDCOMM"
KEYBRK = "BREAK"
NLN = "\n"
DKEYINOUT = { # Out/in, horizontal/vertical or row/col first.
("out",False): KEYROW,
("in",False): KEYCOL,
("out",True): KEYCOL,
("in",True): KEYROW,
}
fidentity = lambda x: x
ffloatd = lambda c: (lambda x: floatdef(x,c))
fspace = lambda x: " {} ".format(x)
fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

def floatdef(x, vdef):
    """Attempt conversion to float, use default value on error.    
    Mainly for empty ratios, double commas.
    """
    try:
        return float(x)
    except ValueError:
        print("'{}' is not a number, converted to {}".format(x,vdef))
        return vdef

class Region():
    """Specific Region used to split a layer to single prompts."""
    def __init__(self, st, ed, base, breaks):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.start = st # Range for the cell (cols only).
        self.end = ed
        self.base = base # How much of the base prompt is applied (difference).
        self.breaks = breaks # How many unrelated breaks the prompt contains.

class Row():
    """Row containing cell refs and its own ratio range."""
    def __init__(self, st, ed, cols):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.start = st # Range for the row.
        self.end = ed
        self.cols = cols # List of cells.
        
def is_l2(l):
    return isinstance(l[0],list) 

def l2_count(l):
    cnt = 0
    for row in l:
        cnt + cnt + len(row)
    return cnt

def list_percentify(l):
    """
    Convert each row in L2 to relative part of 100%. 
    Also works on L1, applying once globally.
    """
    lret = []
    if is_l2(l):
        for row in l:
            # row2 = [float(v) for v in row]
            row2 = [v / sum(row) for v in row]
            lret.append(row2)    #也是二维数组，这里是[[1.0], [1.0]]
    else:
        row = l[:]
        # row2 = [float(v) for v in row]
        row2 = [v / sum(row) for v in row]
        lret = row2    #这里是一维数组[0.4, 0.6]
    return lret

def list_cumsum(l):
    """
    Apply cumsum to L2 per row, ie newl[n] = l[0:n].sum .
    Works with L1.
    Actually edits l inplace, idc.
    """
    lret = []
    if is_l2(l):
        for row in l:
            for (i,v) in enumerate(row):
                if i > 0:
                    row[i] = v + row[i - 1]
            lret.append(row)    #接着上面做成相加型
    else:
        row = l[:]
        for (i,v) in enumerate(row):
            if i > 0:
                row[i] = v + row[i - 1]
        lret = row    #一样接上面做成相加型[0.4, 1.0]
    return lret

def list_rangify(l):
    """
    Merge every 2 elems in L2 to a range, starting from 0.  
    """
    lret = []
    if is_l2(l):
        for row in l:
            row2 = [0] + row
            row3 = []
            for i in range(len(row2) - 1):
                row3.append([row2[i],row2[i + 1]]) 
            lret.append(row3)    #加0变成两两递进(升维)
    else:
        row2 = [0] + l
        row3 = []
        for i in range(len(row2) - 1):
            row3.append([row2[i],row2[i + 1]]) 
        lret = row3    #加0变成两两递进(升维)
    return lret

def ratiosdealer(split_ratio2,split_ratio2r):
    split_ratio2 = list_percentify(split_ratio2)
    split_ratio2 = list_cumsum(split_ratio2)
    split_ratio2 = list_rangify(split_ratio2)
    split_ratio2r = list_percentify(split_ratio2r)
    split_ratio2r = list_cumsum(split_ratio2r)
    split_ratio2r = list_rangify(split_ratio2r)
    return split_ratio2,split_ratio2r

def round_dim(x,y):
    """Return division of two numbers, rounding 0.5 up.    
    Seems that dimensions which are exactly 0.5 are rounded up - see 680x488, second iter.
    A simple mod check should get the job done.
    If not, can always brute force the divisor with +-1 on each of h/w.
    """
    return x // y + (x % y >= y // 2)       

def keyconverter(self,split_ratio,usebase):
    '''convert BREAKS to ADDCOMM/ADDBASE/ADDCOL/ADDROW'''
    #print("\n开始keyconverter!!!")
    if SPLROW not in split_ratio: # Commas only - interpret as 1d.
        #print("split_ratio没有分号")
        split_ratio2 = split_l2(split_ratio, SPLROW, SPLCOL, map_function = ffloatd(1))
        split_ratio2r = [1]
    else:
        #print("有分号，从这里开始")
        (split_ratio2r,split_ratio2) = split_l2(split_ratio, SPLROW, SPLCOL, 
                                        indsingles = True, map_function = ffloatd(1))    #split_ratio2r分行，split_ratio2分列
    #print("**split_ratio2r",split_ratio2r)
    #print("**split_ratio2",split_ratio2)
    (split_ratio2,split_ratio2r) = ratiosdealer(split_ratio2,split_ratio2r)
    #print("***最终的split_ratio2r***:",split_ratio2r)
    #print("***最终的split_ratio2***:",split_ratio2)
    txtkey = fspace(DKEYINOUT[("in", False)]) + NLN
    #print("**txtkey",txtkey)
    lkeys = [txtkey.join([""] * len(cell)) for cell in split_ratio2]
    #print("**lkeys",lkeys)
    txtkey = fspace(DKEYINOUT[("out", False)]) + NLN
    #print("**txtkey",txtkey)
    template = txtkey.join(lkeys)
    #print("template",template)
    #print("**usebase",usebase)
    if usebase:
        template = fspace(KEYBASE) + NLN + template
    #print("template",template)
    changer = template.split(NLN)
    changer = [l.strip() for l in changer]
    keychanger=changer[:-1]    #去掉最后一个换行符
    #print("**keychanger",keychanger)
    for change in keychanger:
        if change == KEYBASE and KEYBASE in self.prompt:    #没有重复的KEYBASE
            continue
        self.prompt= self.prompt.replace(KEYBRK,change,1)                        
    #print("***替换后结果self.prompt***::",self.prompt)

def split_l2(s, key_row, key_col, indsingles = False, map_function = fidentity, split_struct = None):
    lret = []
    if split_struct is None:    #简单拆分逻辑
        #print("split_struct是None，又从这里开始")
        lrows = s.split(key_row)    #分完行得['0.4', '0.6']
        #print("**lrows",lrows)
        lrows = [row.split(key_col) for row in lrows]    #行列都分完 [['0.4'], ['0.6']]
        #print("**lrows",lrows)
        for r in lrows:    #这里遍历行
            cell = [map_function(x) for x in r]    #一般转换成浮点数,文字里面的BREAK当然是0了
            lret.append(cell)
        #print("*lret",lret)
        if indsingles:    #文字九不进这里了
            #print("第二次就不进来了")
            lsingles = [row[0] for row in lret]
            lcells = [row[1:] if len(row) > 1 else row for row in lret]
            #print("**lsingles",lsingles)
            #print("**lcells",lcells)
            lret = (lsingles,lcells)    #一个一维一个二维
    else:
        lrows = str(s).split(key_row)    #按行分开来['0.5']
        #print("*lrows",lrows)
        r = 0
        lcells = []
        lsingles = []
        vlast = 1    #把每行的最后一列捞出来
        for row in lrows:
            row2 = row.split(key_col)
            row2 = [map_function(x) for x in row2]    #[0.5]
            #print("*row2",row2)
            vlast = row2[-1]    #row2是一个一维数组
            #print("*vlast",vlast)
            indstop = False
            while not indstop:
                if (r >= len(split_struct) # Too many cell values, ignore.    #这里说的是行数
                or (len(row2) == 0 and len(split_struct) > 0)): # Cell exhausted.    #struct是[[0, 0, 0], [0, 0, 0]]
                    indstop = True
                if not indstop:
                    if indsingles: # Singles split.
                        lsingles.append(row2[0]) # Row ratio.    #这里是单独的一行里的头一个
                        if len(row2) > 1:
                            row2 = row2[1:]    #去掉第一个取剩下的
                    if len(split_struct[r]) >= len(row2): # Repeat last value.
                        indstop = True
                        broadrow = row2 + [row2[-1]] * (len(split_struct[r]) - len(row2))    #[0.33, 0.33, 0.33]长度一样没有动
                        r = r + 1
                        lcells.append(broadrow)
                    else: # Overfilled this row, cut and move to next.
                        broadrow = row2[:len(split_struct[r])]
                        row2 = row2[len(split_struct[r]):]
                        r = r + 1
                        lcells.append(broadrow)
        # If not enough new rows, repeat the last one for entire base, preserving structure.
        cur = len(lcells)
        while cur < len(split_struct):
            lcells.append([vlast] * len(split_struct[cur]))
            cur = cur + 1
        lret = lcells
        if indsingles:
            lsingles = lsingles + [lsingles[-1]] * (len(split_struct) - len(lsingles))
            lret = (lsingles,lcells)
        #print("***lret***::",lret)
    return lret
    
def matrixdealer(self, split_ratio, baseratio):
    #print("\n开始matrixdealer!!!")
    prompt = self.prompt    #这是之前规范化后的prompt
    #print("**prompt",prompt)
    if KEYBASE in prompt: prompt = prompt.split(KEYBASE,1)[1]    #去掉base后的prompt
    #print("**prompt",prompt)
    if (KEYCOL in prompt.upper() or KEYROW in prompt.upper()):
        breaks = prompt.count(KEYROW) + prompt.count(KEYCOL) + int(self.usebase)    #分行分列base的中断次数
        #print("**prompt",prompt)
        lbreaks = split_l2(prompt, KEYROW, KEYCOL, map_function = fcountbrk)    #lbreaks就记录每部分prompt里面的break数量？
        #print("***breaks***::",breaks)
        #print("***lbreaks***::",lbreaks)
        if (SPLROW not in split_ratio and (KEYROW in prompt.upper()) != (KEYCOL in prompt.upper())):    #split_ratio就是最初的,prompt是标准化去头的
            #print("真")
            # By popular demand, 1d integrated into 2d.
            # This works by either adding a single row value (inner),
            # or setting flip to the reverse (outer).
            # Only applies when using just ADDROW / ADDCOL keys, and commas in ratio.
            split_ratio = "1" + SPLCOL + split_ratio
            #print("split_ratio休整前::",split_ratio)
            (split_ratio2r,split_ratio2) = split_l2(split_ratio, SPLROW, SPLCOL, indsingles = True,
                                map_function = ffloatd(1), split_struct = lbreaks)
            #print("修整后split_ratio2r::",split_ratio2r)
            #print("休整后split_ratio2::",split_ratio2)
        else: # Standard ratios, split to rows and cols.
            #print("条件假")
            (split_ratio2r,split_ratio2) = split_l2(split_ratio, SPLROW, SPLCOL, indsingles = True,
                                            map_function = ffloatd(1), split_struct = lbreaks)    #加上split结构执行一次,按结构整改
            #print("***split_ratio2r后***::",split_ratio2r)
            #print("***split_ratio2后***::",split_ratio2)
        # More like "bweights", applied per cell only.
        baseratio2 = split_l2(baseratio, SPLROW, SPLCOL, map_function = ffloatd(0), split_struct = lbreaks)    #把baseprompt分发到没一格子
    (split_ratio,split_ratior) = ratiosdealer(split_ratio2,split_ratio2r)    #规范化
    #print("***split_ratio***::",split_ratio)
    #print("***split_ratior***:",split_ratior)
    baseratio = baseratio2
    #print("baseratio",baseratio)
    
    # Merge various L2s to cells and rows.
    #print("*******************************************************************************************************\n")
    drows = []
    for r,_ in enumerate(lbreaks):
        dcells = []
        for c,_ in enumerate(lbreaks[r]):
            #print("r",r)
            #print("c",c)
            #print(split_ratio)
            d = Region(split_ratio[r][c][0], split_ratio[r][c][1], baseratio[r][c], lbreaks[r][c])
            #print("start::",d.start,"end::",d.end,"base::",d.base,"breaks::",d.breaks)
            dcells.append(d)
        #print("ooo",split_ratior[r][0],split_ratior[r][1])
        drow = Row(split_ratior[r][0], split_ratior[r][1], dcells)
        #print("row::","start::",drow.start,"end",drow.end)
        #print("\n")
        drows.append(drow)
    #print("***",self.split_ratio)
    self.split_ratio = drows
    #print("***",self.split_ratio[0].start,self.split_ratio[0].end,self.split_ratio[0].cols[0].start,self.split_ratio[0].cols[0].end,self.split_ratio[0].cols[0].base)
    self.baseratio1 = baseratio
    #print("********************************************************************************************************")
    #print("self.split_ratio",self.split_ratio)
    #print("self.base_ratio",baseratio)

# class test:
#     def __init__(self, prompt,split_ratio=None,baseratio=0.2,usebase=False):
#         self.prompt = prompt
#         self.split_ratio = split_ratio
#         self.baseratio = 0.2
#         self.usebase = usebase
# test_prompt='a girl BREAK a cute boy BREAK a dog BREAK a tree.'
# split_ratio='1,1,1;1,1,1'
# x=test(test_prompt,split_ratio)
# keyconverter(x,split_ratio,usebase=False)
# print(x.prompt)
# matrixdealer(x, split_ratio, 0.2)

