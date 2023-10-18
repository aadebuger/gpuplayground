from transformers import pipeline

import os
mdir=os.getenv("mdir")
image_to_text = pipeline("image-to-text", model=f"{mdir}/nlpconnect/vit-gpt2-image-captioning")

r=image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")
print(r)