# Automatic Map Storytelling with Generative Pre-trained Transformer (GPT) Models
## Description
Historical maps provide valuable information and knowledge about the past. However, as they often feature non-standard projections, hand-drawn styles, and artistic elements, it is challenging for non-experts to identify and interpret them. While existing image captioning methods have achieved remarkable success on natural images, their performance on maps is suboptimal as maps are underrepresented in their pre-training process. Also, the granularity of the generated captions is often too rough to describe maps precisely—people should not only learn about the depicted location (_where_) and topic (_what_), but also when and why a map was created. 


To address these problems, we fine-tune the state-of-the-art vision-language model [CLIP](https://github.com/openai/CLIP?tab=readme-ov-file) (Contrastive Language-Image Pre- Training) to generate keyword captions relevant for historical maps and enrich the captions with GPT (Generative Pre-trained Transformer) to tell a map’s story regarding _where_, _what_, _when_ and _why_. We make use of a decision tree to structure and control the granularity of captions with respect to different map types. Our experiment reveals that we can easily fine-tune the base CLIP models to correctly predict map-related keywords and generate detailed and comprehensive captions effectively. In addition, the proposed decision tree architecture can be flexibly extended to other map types and scaled to a larger map captioning system.



## Approach

![OverviewDiagram](https://github.com/claudaff/automatic-map-storytelling/assets/145538566/45cdf01f-e1fe-4660-82b0-0edbcc39d369)






We downloaded maps and their metadata from the online map repository [David Rumsey Map Collection](https://www.davidrumsey.com/). For training, we processed the metadata and maps to generate ground-truth captions to then fine-tune separate CLIP models. During inference, the input map follows the structure of our proposed decision tree, where at the tree’s root node, the map type is determined first. Then, keyword captions with respect to this map type are generated. At last, a GPT model is used to merge the generated keywords and extend the map’s story about the _why_ component. Building on that, we developed a web application for interactive map storytelling.

## Reproduction
Step by step instructions to reproduce our results with our proposed approach.
### 1. Training prerequisites

```sh
git clone https://github.com/rmokady/automatic-map-storytelling && cd automatic-map-storytelling
conda env create -f environment.yml
conda activate map_storytelling
```

### 2. Map datasets

Download and unzip the following fifteen .zip files containing our collected maps with associated metadata (1.6 GB overall).

[M1](https://drive.google.com/file/d/1EWVyhGqqPq-9bQUSOFxBd-L3zaVjfbbl/view?usp=drive_link), 
[M2](https://drive.google.com/file/d/1ZV-0CT_9Nh21yLHyajoVsGyZKywo03UB/view?usp=drive_link)
[M3](https://drive.google.com/file/d/11XBnAgegMf-jWNlMAStL4w_U3CWCuAD5/view?usp=drive_link), 
[M4](https://drive.google.com/file/d/1SoZGjEao8B0j9B0kBu79GxsUMg-gjCW1/view?usp=drive_link), 
[M5](https://drive.google.com/file/d/1FGNIDbX1Js5Wjv7vaRUy6PRo7-bD2D0K/view?usp=drive_link), 
[M6](https://drive.google.com/file/d/1GT6Ulfr1cR9CXuTbfXLKqzkokD00MV8z/view?usp=drive_link), 
[M7](https://drive.google.com/file/d/14_u9gn3nwjOQHaokB9gT-dV8nYF5YMOW/view?usp=drive_link), 
[M8](https://drive.google.com/file/d/1xjyaI4xaKWzk1ODERfAwMFhhUIWw1deM/view?usp=drive_link), 
[M9](https://drive.google.com/file/d/1nBRwbnYcDk4feWYCSXtEUh3qVrfmdA7l/view?usp=drive_link), 
[M10](https://drive.google.com/file/d/1S7NFe8zjyOH3IMWFtQH8EzseE0VIQSm4/view?usp=drive_link), 
[M11](https://drive.google.com/file/d/1o3XjaPnexo0ZUh2kB-HVLCsgxMJzBkeF/view?usp=drive_link), 
[M12](https://drive.google.com/file/d/1C3KnB_P9XAyn2ou6Vb3KuvMzszCTvGN0/view?usp=drive_link), 
[M13](https://drive.google.com/file/d/1i3REduWyjhef9lXF6RuWuWIvSDif-Gxz/view?usp=drive_link), 
[M14](https://drive.google.com/file/d/1dcXKBu4rgtkZXJSOhpGYnpA43UrCwj_5/view?usp=drive_link), 
[M15](https://drive.google.com/file/d/1H_4D-I1EKuF8ggXIRLNjxQkf-GJQExot/view?usp=drive_link)

### 3. Generate ground-truth captions

Run the two scripts `CaptionGenerationClassical.py` (for topographic maps) and `CaptionGenerationPictorial.py` (for pictorial maps). The output will be two NumPy arrays (one containing the map image paths and one containing the corresponding ground-truth captions) for each of the six caption categories. 

### 4. CLIP fine-tuning

Run the six fine-tuning scripts `fineTuneCLIP{Caption Category}`. The output will be six fine-tuned CLIP models. One for each caption category.

Alternatively, download the six fine-tuned models here (3.4 GB overall):

[FT1](https://drive.google.com/file/d/1SAH4cqQSmvywsvNloYLlopn5EAiHbWrR/view?usp=drive_link), 
[FT2](https://drive.google.com/file/d/1d-oyhA2NjpKWyXV2J8C9e9SOIJ9eeRyp/view?usp=drive_link), 
[FT3](https://drive.google.com/file/d/1N37UD8fBmicv3dXnqB3VvWMpuGH641XK/view?usp=drive_link), 
[FT4](https://drive.google.com/file/d/1ln04Twd3tXXON5WNIMPvBaG-3T7ZSDlw/view?usp=drive_link), 
[FT5](https://drive.google.com/file/d/1AGL_WaqzjWNGwLUpuj8Mn346F5SLEMP6/view?usp=drive_link), 
[FT6](https://drive.google.com/file/d/13gb1JBve4er4AGR8HgdEijNVmgeAj291/view?usp=drive_link)

### 5. Inference

1. Download our test maps here (less than 50 MB) and unzip: [Pictorial Test Maps](https://drive.google.com/file/d/1LyYpksg86X1TLUb5LKfSTAD7aCQ_RE68/view?usp=drive_link), [Topographic Test Maps](https://drive.google.com/file/d/1C7O-Jp8Y92nJ8dgkazp44yVbzzqs1_RL/view?usp=drive_link) 


2. Run the script `Inference.py` after reading the instructions in the comments. This script allows testing the six fine-tuned models separately on our test maps.

## Map Storytelling GUI

To run our map storytelling web app, open the script `CaptionInferenceGUI.py`, add your own OpenAI API Key and run it. Make sure that the six fine-tuned models (FT1 to FT6) were downloaded.

Alternatively, if no API Key is available a 'light' version of our approach can be tested without GPT.
For this open `CaptionInferenceLight.py` and assign `input_map` the path to the desired historical map. Running this script will generate corresponding keyword captions with no "Why?" part. 


