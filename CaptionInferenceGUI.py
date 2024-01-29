import gradio as gr
import torch
import clip
from PIL import Image
import numpy as np
import torch.nn as nn
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# =====================OpenAI===========================
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key='*************************************',
)

class Combined_model(nn.Module):
    def __init__(self, model_maptype, model_location, model_century, model_note, model_area, model_topic):
        super(Combined_model, self).__init__()
        self.model_maptype = model_maptype
        self.model_location = model_location
        self.model_century = model_century
        self.model_note = model_note
        self.model_area = model_area
        self.model_topic = model_topic

    def forward(self, x):
        maptypes = ["topographic map", "pictorial map"]
        text = clip.tokenize(maptypes).to(device)
        logits_per_image, logits_per_text = self.model_maptype(x, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        maptype = maptypes[np.argmax(probs)]

        if maptype == "topographic map":
            locations = ["greece", "italy", "iberian peninsula", "france", "eastern hemisphere", "europe",
                         "middle east", "asia minor", "germany", "british isles", "world", "egypt", "part of italy",
                         "part of france", "part of germany", "india", "holy land", "asia", "caucasus", "sri lanka",
                         "south america", "americas", "switzerland", "scandinavia", "netherlands", "africa",
                         "part of greece"]
            text = clip.tokenize(locations).to(device)
            logits_per_image, logits_per_text = self.model_location(x, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            location = locations[np.argmax(probs)]

            centuries = ["19th century", "18th century", "17th century", "16th century"]
            text = clip.tokenize(centuries).to(device)
            logits_per_image, logits_per_text = self.model_century(x, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            century = centuries[np.argmax(probs)]

            notes = ["hand colored", "hand colored with decorative elements and pictorial relief", "pictorial relief", "hand colored with pictorial relief", "engraved", "decorative elements and pictorial relief"]
            text = clip.tokenize(notes).to(device)
            logits_per_image, logits_per_text = self.model_note(x, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            note = notes[np.argmax(probs)]

            return maptype, location, century, note

        elif maptype == "pictorial map":
            areas = ["united states", "world"]
            text = clip.tokenize(areas).to(device)
            logits_per_image, logits_per_text = self.model_area(x, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            area = areas[np.argmax(probs)]

            topics = ['flight network', 'news during world war 2', 'world war 2', 'transport routes', 'tourist sights', 'playing card', 'satirical representation', 'people', 'educational drawings', 'food and agriculture', 'animals', 'military', 'stamps']
            text = clip.tokenize(topics).to(device)
            logits_per_image, logits_per_text = self.model_topic(x, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            topic = topics[np.argmax(probs)]

            return maptype, area, topic

model_maptype = copy.deepcopy(model)
model_location = copy.deepcopy(model)
model_century = copy.deepcopy(model)
model_note = copy.deepcopy(model)
model_area = copy.deepcopy(model)
model_topic = copy.deepcopy(model)

def freeze_network(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

model_path_maptype = "CLIPMapType.pt"
model_maptype.load_state_dict(torch.load(model_path_maptype, map_location=device))
freeze_network(model_maptype)

model_path_location = "CLIPLocationTopo.pt"
model_location.load_state_dict(torch.load(model_path_location, map_location=device))
freeze_network(model_location)

model_path_century = "CLIPCentury.pt"
model_century.load_state_dict(torch.load(model_path_century, map_location=device))
freeze_network(model_century)

model_path_note = "CLIPStyle.pt"
model_note.load_state_dict(torch.load(model_path_note, map_location=device))
freeze_network(model_note)

model_path_area = "CLIPLocationPict.pt"
model_area.load_state_dict(torch.load(model_path_area, map_location=device))
freeze_network(model_area)

model_path_topic = "CLIPTopic.pt"
model_topic.load_state_dict(torch.load(model_path_topic, map_location=device))
freeze_network(model_topic)


# ===================interface of GUI========================
def map_interface(map, what, where, when, why):
    if type(map) == "str":
        image = preprocess(Image.open(map)).unsqueeze(0).to(device)
    else:
        map = Image.fromarray(map)
        image = preprocess(map).unsqueeze(0).to(device)

    results = []
    combined_model = Combined_model(model_maptype, model_location, model_century, model_note, model_area, model_topic)
    combined_model.eval()
    with torch.no_grad():
        results = combined_model(image)

    prompt = ""
    if what:
        prompt += f"What is this map about? "
    if where:
        prompt += f"Where is this map about? "
    if when:
        prompt += f"When is this map about? "
    if why:
        prompt += f"What can this map be used for? "


    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant"},
                {"role": "user",
                 "content": f"Please create a concise sentence that encapsulates these keywords: {results}. Additionally, provide a brief explanation, in under 30 words, about: {prompt}."
                 }
            ],
            max_tokens=60
        )
    results = response.choices[0].message.content
    results = results.strip('"')

    return results



with gr.Blocks() as demo:
    with gr.Tab("Demo"):
        with gr.Row("Map Details"):
            with gr.Column("Map"):
                image_input = gr.Image(label="Upload or Drag Map Here", type='numpy')
                with gr.Row("Map Details"):
                    what = gr.Checkbox(label="What")
                    where = gr.Checkbox(label="Where")
                    when = gr.Checkbox(label="When")
                    why = gr.Checkbox(label="Why")

                submit_button = gr.Button("Submit")

            output_text = gr.Textbox(label="Caption")

        # Define the interaction
        submit_button.click(
            fn=map_interface,
            inputs=[image_input, what, where, when, why],
            outputs=output_text
        )

    with gr.Tab("README"):
        gr.Markdown("""
            # Map Storytelling Tool

            This demo application utilizes the `CLIP` model and `OpenAI`'s GPT-3.5 model to analyze uploaded map images and generate relevant descriptions as storytelling.

            ## Features

            - **Map Type Recognition**: The app can identify the type of the uploaded map (e.g., topographic or pictorial).
            - **Map Details Extraction**: Based on the type of map, the app will identify specific details (such as region, theme, date, etc.).
            - **Intelligent Description Generation**: Uses the GPT-3.5 model to generate descriptions of the map based on identified information.

            ## How to Use

            1. **Upload a Map**: Upload a map image by clicking or dragging.
            2. **Select Details**: Choose the map details you wish to analyze (e.g., location, time, purpose).
            3. **Generate Description**: Click the "Submit" button and wait for the system to process and generate a description of the map.

            ## Technical Background

            This tool combines cutting-edge image recognition and natural language processing technologies to provide accurate map analysis and description generation.

            ## Notes

            - Ensure that the uploaded map is clear for the system to accurately recognize.
            - The generation of descriptions may take a few seconds to process.

            """)

if __name__ == "__main__":
    demo.launch(share=True)
