import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Python script to test fine-tuned CLIP models per caption category on test maps

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# paths to test maps: 
# "testmap_paths.npy" contains both topographic and pictorial maps
# "testmap_pathsTopographic.npy" contains only topographic maps
# "testmap_pathsPictorial.npy" contains only pictorial maps

test_image_paths = np.load("testmap_paths.npy")  

# ground-truth captions:
# "testmap_MapType.npy" -> use in only with "testmap_paths.npy" as paths to predict the map type
# "testmap_LocationTopographic.npy" -> use only with "testmap_pathsTopographic.npy" to predict topographic map locations
# "testmap_CenturyTopographic.npy" -> use only with "testmap_pathsTopographic.npy" to predict the century
# "testmap_LocationPictorial.npy" -> use only with "testmap_pathsPictorial.npy" to predict pictorial map locations

test_captions = np.load("testmap_MapType.npy") 

# Uncomment following line to  later visually assess the predictions for categories 'Style' or 'Topic'
# test_captions = np.zeros(113) # acts as placeholder as placeholder as no test map ground-truth exists for these categories
# Use "testmap_pathsTopographic.npy" for 'Style' and "testmap_pathsPictorial.npy" for 'Topic' as paths!

print(len(test_captions))
print(len(test_image_paths))

useFineTuned = True # False = Use base CLIP model; True = Use chosen fine-tuned CLIP model

if useFineTuned:

    model_path = "CLIPMapType.pt" # Replace with corresponding model!
    
    # Map Type -> "CLIPMapType.pt"
    # Location (topographic) -> "CLIPLocationTopo.pt"
    # Style -> "CLIPStyle.pt"
    # Century -> "CLIPCentury.pt"
    # Location (pictorial) -> "CLIPLocationPict.pt"
    # Topic -> "CLIPTopic.pt"
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    model = model.to(device)
    print("Using Fine-tuned model")

# uncomment 'classes' list for corresponding model, comment remaining lists.
# e.g., for Map Type uncomment 'classes = ["topographic map", "pictorial map"]' and comment the other lists

# Map Type
classes = ["topographic map", "pictorial map"]

# Location (topographic)
# classes = ["greece", "italy", "iberian peninsula", "france", "eastern hemisphere", "europe", "middle east", "asia minor", "germany", "british isles", "world", "egypt", "part of italy", "part of france", "part of germany", "india", "holy land", "asia", "caucasus", "sri lanka",  "south america", "americas", "switzerland", "scandinavia", "netherlands", "africa", "part of greece"]

# Century
# classes = ["19th century", "18th century", "17th century", "16th century"]

# Style
# classes = ["hand colored", "hand colored with decorative elements and pictorial relief", "pictorial relief", "hand colored with pictorial relief", "engraved", "decorative elements and pictorial relief"]

# Topic
# classes = ['flight network', 'news during world war 2', 'world war 2', 'transport routes', 'tourist sights', 'playing card', 'satirical representation', 'people', 'educational drawings', 'food and agriculture', 'animals', 'military',  'stamps']

# Location (pictorial)
# classes = ["united states", "world"]

# ------------------------------------------------------------------------------

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
    # plt.show() # uncomment to view image

    total += 1

    if prediction == groundtruth:
        count += 1

print("Accuracy: ", count / total) # calculate prediction accuracy 
