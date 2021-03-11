# facebook's blenderbot
# A state-of-the-art open source chatbot
from transformers import BlenderbotTokenizer, TFBlenderbotForConditionalGeneration
mname = 'facebook/blenderbot-400M-distill'
model = TFBlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

import os
clear = lambda: os.system('cls')
clear()

while True :
    UTTERANCE = input("User:  ")
    if UTTERANCE == 'quit':
        break
    inputs = tokenizer([UTTERANCE], return_tensors='tf')
    reply_ids = model.generate(**inputs)
    print("Bot : ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
