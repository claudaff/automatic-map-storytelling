import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Python script to test fine-tuned CLIP models per caption category on test maps

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

test_image_paths = np.load("testmap_paths.npy")  # paths to test maps
test_captions = np.load("testmap_captions.npy")  # corresponding captions, if available (here: map type)

print(len(test_captions))
print(len(test_image_paths))

useFineTuned = True # False = Use base CLIP model; True = Use chosen fine-tuned CLIP model

if useFineTuned:

    model_path = "checkpoints/best_model_MapType.pt" # chosen model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    model = model.to(device)
    print("Using Fine-tuned model")

# uncomment 'classes' list for corresponding model, comment remaining lists

classes = ["topographic map", "pictorial map"]
# classes = ["greece", "italy", "iberian peninsula", "france", "eastern hemisphere", "europe", "middle east", "asia minor", "germany", "british isles", "world", "egypt", "part of italy", "part of france", "part of germany", "india", "holy land", "asia", "caucasus", "sri lanka",  "south america", "americas", "switzerland", "scandinavia", "netherlands", "africa", "part of greece"]
# classes = ["19th century", "18th century", "17th century", "16th century"]
# classes = ["hand colored", "hand colored with decorative elements and pictorial relief", "pictorial relief", "hand colored with pictorial relief", "engraved", "decorative elements and pictorial relief"]
# classes = ['flight network', 'news during world war 2', 'world war 2', 'transport routes', 'tourist sights', 'playing card', 'satirical representation', 'people', 'educational drawings', 'food and agriculture', 'animals', 'military',  'stamps']
# classes = ["united states", "world"]

print("Number of classes: ", len(classes))
count = 0
total = 0
for testmap, groundtruth in zip(test_image_paths, test_captions):

    image = preprocess(Image.open(testmap)).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print("Label probs:", probs)
    prediction = classes[np.argmax(probs)] # prediction is class with highest probability
    print(prediction)
    print("Ground Truth:", groundtruth)
    print(testmap)
    img = np.asarray(Image.open(testmap))
    plt.imshow(img)
    # plt.show()

    total += 1

    if prediction == groundtruth:
        count += 1

print("Accuracy: ", count / total) # calculate prediction accuracy 
