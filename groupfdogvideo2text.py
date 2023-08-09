import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import openai


def video2text(mp4_file):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load pretrained processor, tokenizer, and model
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    # load video
    ## video_path = "DogAction3.mp4"
    container = av.open(mp4_file)
    "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"
    # extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
      if i in indices:
          frames.append(frame.to_ndarray(format="rgb24"))

    # generate caption
    gen_kwargs = {
      "min_length": 10,
      "max_length": 20,
      "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return caption


def chatgpt(caption, final_emotion, api_key="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"):
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=500,
        messages=[
            {"role": "system", "content": 'You are my dog and I am your owner'},
            {"role": "user", "content": f'You are a {final_emotion} dog and you are in the scenario that {caption}. Please use one or two sentence to describe the feelings in your heart right now, and respond using a first-person perspective.'}
        ]
    )
    answer = completion.choices[0].message.content
    return answer

