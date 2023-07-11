#from flask import Flask, request

#app = Flask(__name__)


#@app.route("/schmidt", methods=['POST', "GET"])
#def login():
#    print(request.method)
#    return f"Received message {request}"


#print("main.py executed successfully.")

#if __name__ == "__main__":
#    app.run(debug=True, port=12345)

from flask import Flask, request
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch, uuid, os
from PIL import Image

app = Flask(__name__)
ROOT = r"./images"

@app.route('/python-backend', methods=["POST"])
def request_handler():
    data = request.data
    for filename in request.files:
        file = request.files[filename]
        if file:
            save_filename = os.path.join(ROOT, str(uuid.uuid4())+".jpg")
            file.save(save_filename)
    if os.path.isfile(save_filename):
        text = predict_step([save_filename])[0]
        return text
    else:
        return "Shit\n"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


#predict_step(['football-match.jpg', 'cat-3.jpg']) # ['a woman in a hospital bed with a woman in a hospital bed']



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1880)
