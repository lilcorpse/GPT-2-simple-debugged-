import gpt_2_simple as gpt2
import os
import requests
import io
import sys
import codecs


model_name = "345M"
if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "C:\\Users\\enrique\\gpt-2\\src\\chp30.txt"
if not os.path.isfile(file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
                pass
                f.write(file_name.encode('cp1252').decode('cp1252').encode("utf-8"))

    
sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=1000)   # steps is max number of training steps

gpt2.generate(sess)
