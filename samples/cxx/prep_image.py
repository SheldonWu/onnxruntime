

from PIL import Image
import os.path as osp

## fns_candy_style_transfer
inp_imgs=["/workspace/model/c001.png","/workspace/model/c002.jpg"]
(w_fns_inp, h_fns_inp) = (720,720)
for i_img in inp_imgs:
    image = Image.open(i_img)
    new_image = image.resize((w_fns_inp, h_fns_inp))
    (d, file_name) = osp.split(i_img)
    new_image.save(osp.join(d, osp.splitext(file_name)[0]+"_i.png"))
