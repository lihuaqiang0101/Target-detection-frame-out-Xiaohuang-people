from PIL import Image
import random
import numpy as np
import os
def Paste(img1,img2):
    # 加载底图
    base_img = Image.open(os.path.join('datasets',img1))
    base_img = base_img.resize((224,224))
    # 可以查看图片的size和mode，常见mode有RGB和RGBA，RGBA比RGB多了Alpha透明度
    # print base_img.size, base_img.mode
 
    # 加载需要P上去的图片
    tmp_img = Image.open(os.path.join('yellow',img2))
    try:
         # 底图上需要P掉的区域
        region = tmp_img
        # 使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
        # 注意，region的大小必须和box的大小完全匹配。但是两张图片的mode可以不同，合并的时候回自动转化。如果需要保留透明度，则使用RGMA mode
        # 提前将图片进行缩放，以适应box区域大小
        new_w = random.randint(50, 100)
        new_h = random.randint(50, 100)
        region.resize((new_w, new_h))
        w = random.randint(0, 224-new_w)
        h = random.randint(0, 224-new_h)
        box = (w, h, w + new_w, h + new_h)  #
        region = region.resize((box[2] - box[0], box[3] - box[1]))
        region = region.rotate(random.randint(-45, 45))  # 对图片进行旋转
        base_img.save('dataset/{}.0.0.0.0.0.png'.format(img1.split('.')[0]))  # 保存图片
        region = region.convert('RGBA')
        b,g,r,a = region.split()
        base_img.paste(region, box,mask=a)
        # base_img.show() # 查看合成的图片
       base_img.save('dataset/{}.{}.{}.{}.{}.1.png'.format(img1.split('.')[0],w,h,w+new_w,h+new_h,1)) #保存图片
    except:
        pass
 
bj = os.listdir('datasets')
bj.sort(key=lambda x:int(x.split('.')[0]))
for img1 in bj:
    qj = os.listdir('yellow')
    for img2 in qj:
        Paste(img1,img2)
