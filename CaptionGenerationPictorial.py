import os
import re
from collections import Counter
import numpy as np
import cv2

# Python script to assign correct topic and area to pictorial maps

maps_directories = [
    "AirlinesWorld",
    "MilitaryWorld",
    "AtlasWorld",
    "ErnestWorld",
    "AgricultureUSA",
    "AirlinesUSA",
    "EthnographyUSA",
    "IndiansUSA",
    "MilitaryUSA",
    "RailroadsUSA",
    "RoadsUSA",
    "SatiricalUSA",
    "TourismUSA",
]  # Folders where sub-folders containing maps (.jpg) and metadata (.text) are stored

image_paths = []  # List to store map paths
captions = []  # List to store captions

path_area = []  # List to store map paths
captions_area = []  # List to store caption concerning the area (World / USA)

for maps_directory in maps_directories:

    # Iterate over the sub-folders in the directory
    for root, dirs, files in os.walk(maps_directory):

        for file in files:

            file_path = os.path.join(root, file)

            # Check if the file is an image (.jpg) or text (.txt) file

            if file.lower().endswith(".jpg"):

                image_paths.append(file_path)

            elif file.lower().endswith(".txt"):

                with open(file_path, "r", encoding="utf-8") as txt_file:

                    txt_content = txt_file.read()

                    short_title = ""
                    world_area = ""
                    subject = ""
                    country = ""
                    date = ""

                    for line in txt_content.split("\n"):
                        if line.startswith("Short Title"):
                            short_title = line.split("\t", 1)[1]

                        elif line.startswith("World Area"):
                            world_area = line.split("\t", 1)[1]

                        elif line.startswith("Country"):
                            country = line.split("\t", 1)[1]

                        elif line.startswith("Date"):
                            date = line.split("\t", 1)[1]
                            date = int(date)

                    short_title = short_title.split(".", 1)[0]  # Remove punctuations

                    # Assign correct area to maps

                    if "box" not in short_title.lower():

                        if ("USA" in maps_directory) or (
                            "Bird and Flower Map" in file_path
                        ):

                            captions_area.append("united states")
                            path_area.append(file_path)

                        elif "World" in maps_directory:

                            captions_area.append("world")
                            path_area.append(file_path)

                    # Assign correct topic to maps

                    if maps_directory == "AirlinesWorld":

                        if re.search(r"Air France", short_title) or re.search(
                            r"Reseau", short_title
                        ):

                            captions.append(f"flight network")

                        else:

                            captions.append(f"flight network")

                    elif maps_directory == "MilitaryWorld":

                        if re.search(r"World News", short_title) or re.search(
                            r"Newsmap", short_title
                        ):

                            captions.append(f"news during world war 2")

                        elif re.search(r"Dated", short_title):

                            captions.append(f"world war 2")

                        else:

                            captions.append(f"world war 2")

                    elif maps_directory == "AtlasWorld":

                        if re.search(r"Playing", short_title):

                            captions.append(f"playing card")

                        elif (
                            re.search(r"lignes", short_title)
                            or re.search(r"Messageries", short_title)
                            or re.search(r"Bremen", short_title)
                        ):

                            captions.append(f"transport routes")

                        elif (
                            re.search(r"Animals", short_title)
                            or re.search(r"Dog", short_title)
                            or re.search(r"Horse", short_title)
                            or re.search(r"animals", short_title)
                            or re.search(r"Gara", short_title)
                            or re.search(r"Zoo", short_title)
                        ):

                            captions.append(f"animals")

                        elif re.search(r"Fauna", short_title) or re.search(
                            r"Fish", short_title
                        ):

                            captions.append(f"animals")

                        elif (
                            re.search(r"Distribution", short_title)
                            or re.search(r"Ciclones", short_title)
                            or re.search(r"Volcanic", short_title)
                            or re.search(r"waters", short_title)
                            or re.search(r"Physical", short_title)
                        ):

                            captions.append(f"BOX")

                        elif re.search(r"The World", short_title) or re.search(
                            r"Earth", short_title
                        ):

                            captions.append(f"educational drawings")

                        elif (
                            re.search(r"Races", short_title)
                            or re.search(r"Pictorial", short_title)
                            or re.search(r"Peoples", short_title)
                        ):

                            captions.append(f"people")

                        elif re.search(r"Cuba", short_title):

                            captions.append(f"BOX")

                        else:

                            captions.append("BOX")

                    elif maps_directory == "ErnestWorld":

                        if (
                            re.search(r"stamps", short_title)
                            or re.search(r"Stamps", short_title)
                            or re.search(r"Stamp", short_title)
                        ):

                            captions.append(f"stamps")

                        elif re.search(r"War", short_title):

                            captions.append(f"world war 2")

                        elif re.search(r"Wonders", short_title):

                            captions.append(f"wonders")

                        else:

                            captions.append(f"planes")

                    elif maps_directory == "AgricultureUSA":

                        captions.append(f"food and agriculture")

                    elif maps_directory == "AirlinesUSA":

                        if re.search(r"American Airlines", short_title):

                            captions.append(f"flight network")

                        elif re.search(r"TWA", short_title):

                            captions.append(f"flight network")

                        else:

                            captions.append(f"flight network")

                    elif maps_directory == "EthnographyUSA":

                        captions.append(f"people")

                    elif maps_directory == "IndiansUSA":

                        captions.append(f"people")

                    elif maps_directory == "MilitaryUSA":

                        captions.append(f"military")

                    elif maps_directory == "RailroadsUSA":

                        captions.append(f"transport routes")

                    elif maps_directory == "RoadsUSA":

                        captions.append(f"transport routes")

                    elif maps_directory == "SatiricalUSA":

                        captions.append(f"satirical representation")

                    elif maps_directory == "TourismUSA":

                        captions.append(f"tourist sights")

                    else:

                        print("Error!")

print(Counter(captions))
print(len(captions))

np.save("areasTEST1.npy", np.array(captions_area))
np.save("pathsTEST1.npy", np.array(image_paths))

np.save("captionsTEST2.npy", np.array(captions))
np.save("pathsTEST2.npy", np.array(image_paths))

# Post-process captions and paths (for topics)

caps = np.load("captionsTEST2.npy")
paths = np.load("pathsTEST2.npy")

print(len(caps))
print(len(paths))

captions = []
map_paths = []

for c, p in zip(caps, paths):

    if "BOX" not in c:

        img = cv2.imread(p)

        try:

            # print(c)
            # print(p)
            captions.append(c)
            map_paths.append(p)
            # cv2.imshow('map', img)

        except:

            cv2.error

        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image


print(Counter(captions))
print("Count- ", len(Counter(captions)))
print(len(captions))
print(len(map_paths))

np.save("PictorialCaptions.npy", captions)
np.save("PictorialPaths.npy", map_paths)
