import json, os, tqdm
import jittor as jt

from JDiffusion.pipelines import StableDiffusionPipeline

max_num = 28
dataset_root = "/root/autodl-tmp/JDiffusion-master/examples/dreambooth/B"
cls = ["in Neon Light line Art Style",  # 0 霓虹灯艺术风格
       "in Impressionist oil painting style",  # 1 印象派风格
       "in Brown yarn art Style",  # 2 黏土雕塑风格
       "in Modern signature pen Line Art Style",  # 3 奇幻线条艺术
       "in Virtual Cloud Art Style",  # 4 拟物云朵艺术
       "in Retro illustration style",  # 5 复古航海幻想
       "in Traditional Chinese Paper Cuttings art Style",  # 6 传统中国剪纸艺术
       "in Modern Urban Illustrations Style",  # 7 现代都市春景插画
       "in Geometric Paper Art Style",  # 8 几何纸艺
       "in Retro watercolor art style",  # 9  水彩插画
       "in Abstract Expressionism Style",  # 10 抽象表现主义
       "in Anime Cartoon Drawing Style",  # 11 卡通风格
       "in pixel 3D art Style",  # 12 像素化3D艺术
       "in Traditional Chinese Ink Painting Style",  # 13 传统中国水墨画
       "in Modern signature pen Line Art Style",  # 14 抽象部落艺术
       "in Cyberpunk Art Style",  # 15 鲜艳的复古未来主义
       "in Pixel pseudo object style",  # 16 像素艺术
       "in Watercolor Landscape Painting Style",  # 17 水彩风景画
       "in Cartoon Fantasy Style",  # 18 卡通奇幻
       "in 3D Pixel Lego Art",  # 19 3D像素艺术
       "in exquisite retro illustration style",  # 20 细致的奇幻插画
       "in Delicate Paper Cuttings art Style",  # 21 精致的剪纸艺术
       "in retro colored lead illustration style",  # 22 复古电器插画
       "in Modern signature pen Line Art Style",  # 23 单色线条艺术
       "in Transparent plastic imitation style",  # 24 写实微缩模型
       "in Cartoon Pixel Art Style",  # 25 像素艺术
       "in Folding Art Style",  # 26 折纸艺术
       "in Retro Futurist Style"]  # 27 复古未来主义蒸汽波
import PIL
import os
with jt.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        pipe.load_lora_weights(f"style/style_{taskid}")

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            print(f"a "+prompt+f" in the style of style_{taskid}, depicted in " +cls[int(taskid)-int('0')])
            #a boat  with  style_j  in  style_i background
            #in a style reminiscent of Vincent van Gogh's Impressionist technique
            #img="/root/autodl-tmp/JDiffusion-master/examples/dreambooth/B/"
            #t=os.listdir(img+str(taskid)+"/images/")[0]
            #image = PIL.Image.open(img+str(taskid)+"/images/"+t)
            image = pipe(f"a "+prompt+f" in the style of style_{taskid}, depicted " +cls[int(taskid)-int('0')], num_inference_steps=75, width=512, height=512).images[0]

            #image = pipe(f"a " + prompt ,
             #            num_inference_steps=25, width=512, height=512).images[0]
            os.makedirs(f"./output/{taskid}/", exist_ok=True)
            image.save(f"./output/{taskid}/{prompt}.png")
