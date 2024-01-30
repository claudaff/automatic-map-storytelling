import os
import numpy as np
from collections import Counter

# Python script to store paths to maps with corresponding ground truth captions

maps_directories = ["ClassicalMaps", "PictorialMaps"]

# Lists to store map paths
image_paths = []
image_paths_class_areas = []
image_paths_class_notes = []
image_paths_class_dates = []
image_paths_class_pic = []

# Set defining which areas have the most usable maps
areas_classical = set(
    (
        "greece",
        "italy",
        "iberian peninsula",
        "france",
        "eastern hemisphere",
        "europe",
        "middle east",
        "asia minor",
        "germany",
        "british isles",
        "world",
        "egypt",
        "part of italy",
        "part of france",
        "part of germany",
        "india",
        "holy land",
        "asia",
        "caucasus",
        "sri lanka",
        "south america",
        "americas",
        "switzerland",
        "scandinavia",
        "netherlands",
        "africa",
        "part of greece",
    )
)

date_set = set("19th century")

image_paths_picTest = []
caps_TEST = []

# Lists to store captions
captions_area = []
captions_date = []
captions_note = []
captions_class_pic = []

area_set = set(())
areas_array = []
dates_array = []
notes_array = []

index = 0
count = 0
count_cities = 0

for maps_directory in maps_directories:

    # Iterate over the sub-folders in the directory
    for root, dirs, files in os.walk(maps_directory):

        for file in files:

            file_path = os.path.join(root, file)

            # Check if the file is an image (.jpg) or text (.txt) file
            if file.lower().endswith(".jpg"):

                image_paths.append(file_path)
                img = image_paths[index]
                index += 1

            elif file.lower().endswith(".txt"):

                with open(file_path, "r", encoding="utf-8") as txt_file:

                    txt_content = txt_file.read()

                    # Variables needed to correctly assign metadata information
                    short_title = ""
                    subject = ""
                    date = ""
                    note = ""
                    event = ""
                    maptype = ""
                    author = ""
                    world_area = ""
                    area = ""
                    country = ""
                    city = ""

                    flagw = True
                    flagc = True
                    flagr = True
                    flags = True

                    count = 0

                    # Scan metadata line for line
                    for line in txt_content.split("\n"):
                        if line.startswith("Note"):
                            note = line.split("\t", 1)[1]

                        elif line.startswith("World Area") and flagw:
                            count += 1
                            flagw = False
                            area = line.split("\t", 1)[1]

                        elif line.startswith("Country") and flagc:
                            count += 1
                            flagc = False
                            area = line.split("\t", 1)[1]

                        elif line.startswith("State/Province") and flags:
                            count += 1
                            flags = False
                            area = line.split("\t", 1)[1]

                        elif line.startswith("Region") and flagr:
                            count += 1
                            flagr = False
                            area = line.split("\t", 1)[1]

                        elif line.startswith("County"):
                            count += 1
                            area = line.split("\t", 1)[1]

                        elif line.startswith("City"):
                            count += 1
                            area = line.split("\t", 1)[1]
                            city = area

                        elif line.startswith("Short Title"):
                            short_title = line.split("\t", 1)[1]

                        elif line.startswith("Type"):

                            maptype = line.split("\t", 1)[1]

                        elif line.startswith("Subject"):

                            subject = line.split("\t", 1)[1]

                        elif line.startswith("Author"):

                            author = line.split("\t", 1)[1]

                        elif line.startswith("Event"):

                            try:

                                event = line.split("\t", 1)[1]

                            except IndexError:

                                event = "none"

                        elif line.startswith("Date"):
                            date = line.split("\t", 1)[1]

                            try:

                                date = int(date)

                            except ValueError:

                                date = int(date.split(".", 1)[0])

                    short_title = short_title.split(".", 1)[0]  # Remove punctuations

                    if maps_directory == "PictorialTest":

                        image_paths_picTest.append(img)

                        caps_TEST.append("pictorial map")

                    if maps_directory == "ClassicalMaps":

                        image_paths_picTest.append(img)
                        caps_TEST.append("topographic map")

                        if isinstance(date, int):

                            if date <= 1600:  # Assign each map the correct century

                                date = "16th century"

                            elif 1600 < date <= 1700:

                                date = "17th century"

                            elif 1700 < date <= 1800:

                                date = "18th century"

                            elif 1800 < date <= 1900:

                                date = "19th century"

                            else:

                                print(date)

                        else:

                            print(date)

                        # Assign correct area based on Short Title or area variable

                        if "4, 5" in short_title:

                            area = "Israel"

                        elif "62" in short_title:

                            area = "Asia Minor"

                        elif (
                            "greece" in short_title.lower()
                            or "grece" in short_title.lower()
                        ):

                            area = "Greece"

                        elif "63" in short_title:

                            area = "Holy Land"

                        elif "India" in short_title:

                            area = "India"

                        elif "Africa Antiqua" in short_title:

                            area = "North Africa"

                        elif "Aegyptus" in short_title:

                            area = "Egypt"

                        elif (
                            "asia minor" in short_title.lower()
                            or "asiae minoris" in short_title.lower()
                        ):

                            area = "Asia Minor"

                        elif "rome" in area.lower():

                            area = "Rome (Italy)"

                        elif "romain, empire" in area.lower():

                            area = "Roman Empire"

                        elif "balkan" in area.lower() or "balkans" in area.lower():

                            area = "Balkans"

                        elif "africa, north" in area.lower():

                            area = "North Africa"

                        elif "europe, central" in area.lower():

                            area = "Central Europe"

                        elif "europe, eastern" in area.lower():

                            area = "Eastern Europe"

                        elif "asia, central" in area.lower():

                            area = "Central Asia"

                        elif "africa, south" in area.lower():

                            area = "Southern Africa"

                        elif "asia, southern" in area.lower():

                            area = "Southern Asia"

                        elif "mecedonia" in area.lower():

                            area = "Macedonia"

                        elif "jerusalem" in area.lower():

                            area = "Jerusalem"

                        if (
                            "alexandri" in short_title.lower()
                            or "alexander" in short_title.lower()
                            or "alexandre" in short_title.lower()
                            or "alessandro" in short_title.lower()
                        ):

                            area = "middle east"

                        if count > 1:  # if more than one area assigned in the metadata

                            if "TAV" in short_title:

                                area = "Rome (Italy)"

                            elif "World" in short_title:

                                area = "World"

                            elif "Babylonia" in short_title:

                                area = "Asia Minor"

                            elif "britain" in short_title.lower():

                                area = "Britain"

                            elif "france" in short_title.lower():

                                area = "France"

                            elif "germany" in short_title.lower():

                                area = "Germany"

                            elif "greece" in short_title.lower():

                                area = "Greece"

                            elif "italy" in short_title.lower():

                                area = "Italy"

                            elif "palestine" in short_title.lower():

                                area = "Palestine"

                            elif "persia" in short_title.lower():

                                area = "Persia"

                            elif (
                                "empire romain" in short_title.lower()
                                or "roman empire" in short_title.lower()
                            ):

                                area = "roman empire"

                            elif "arabia" in short_title.lower():

                                area = "arabian peninsula"

                            elif "belgie" in short_title.lower():

                                area = "Belgium"

                            elif "graecia" in short_title.lower():

                                area = "Greece"

                            elif "itali" in short_title.lower():

                                area = "Italy"

                            elif "pars occidentalis" in short_title.lower():

                                area = "western mediterranean"

                            elif "pars orientalis" in short_title.lower():

                                area = "eastern mediterranean"

                            elif (
                                "palestina" in short_title.lower()
                                or "palaestina" in short_title.lower()
                                or "palestine" in short_title.lower()
                            ):

                                area = "holy land"

                            else:

                                # print(short_title, area)
                                pass
                        else:

                            pass

                        if (
                            "palestine" in area.lower()
                            or "palestina" in area.lower()
                            or "palestine" in short_title.lower()
                        ):

                            area = "Holy land"

                        if "orbis veteribus" in short_title.lower():

                            area = "Eastern	Hemisphere"

                        if "pars occidentalis" in short_title.lower():

                            area = "western mediterranean"

                        if "europe" in short_title.lower():

                            area = "Europe"

                        if (
                            "departement" in short_title.lower()
                            or "governo generale" in short_title.lower()
                            or "map of the geography of ancient france"
                            in short_title.lower()
                        ):

                            area = "Part of France"

                        if area.lower() == "mediterranean region":

                            area = "Mediterranean"

                        if (
                            "herzogthum" in short_title.lower()
                            or "provinz" in short_title.lower()
                        ):

                            area = "Part of Germany"

                        if area.lower() == "portugal":

                            area = "Iberian Peninsula"

                        if "eastern\themisphere" in area.lower():

                            area = "Eastern Hemisphere"

                        if (
                            area.lower() == "roman empire"
                            or "roman empire" in short_title.lower()
                        ):

                            if "peut" in short_title.lower():

                                pass

                            else:

                                area = "Mediterranean"

                        if "spain" in area.lower():

                            area = "iberian peninsula"

                        if "great britain" == area.lower():

                            area = "British isles"

                        if "mediterranean" == area.lower():

                            area = "europe"

                        if "black sea" == area.lower():

                            area = "caucasus"

                        if "america" == area.lower():
                            area = "americas"

                        if " part of germany" == area.lower():
                            area = "part of germany"

                        if " part of greece" == area.lower():
                            area = "part of greece"

                        # Assign correct style information based on Note
                        decorative_element = False

                        if "cartouche" in note.lower() or "vign" in note.lower():

                            decorative_element = True

                        if len(note) == 0:

                            note = "None"

                        elif (
                            "hand colo" in note.lower()
                            or "hand-colo" in note.lower()
                            or ("colo" in note.lower() and "uncolo" not in note.lower())
                        ):

                            if (
                                not decorative_element
                                and "pictorially" not in note.lower()
                            ):

                                note = "Hand colored"

                            elif (
                                decorative_element and "pictorially" not in note.lower()
                            ):

                                note = "Hand colored"  # not including cartouche since too few maps

                            elif (
                                not decorative_element and "pictorially" in note.lower()
                            ):

                                note = "Hand colored with pictorial relief"

                            elif decorative_element and "pictorially" in note.lower():

                                note = "Hand colored with decorative elements and pictorial relief"

                            else:

                                print("ERROR!")

                        elif decorative_element:

                            if "pictorially" in note.lower():

                                note = "decorative elements and pictorial relief"

                            else:

                                note = "decorative elements and pictorial relief"

                        elif "pictorially" in note.lower():

                            note = "pictorial relief"

                        elif "engraved" in note.lower():

                            note = "Engraved"

                        else:

                            note = "None"

                        dates_array.append(date.lower())
                        captions_class_pic.append("topographic map")  # set map type
                        image_paths_class_pic.append(img)

                        captions_date.append(date.lower())
                        image_paths_class_dates.append(img)

                        if "none" not in note.lower():
                            notes_array.append(note.lower())
                            captions_note.append(note.lower())
                            image_paths_class_notes.append(img)

                        areas_array.append((area.lower()))

                        if area.lower() in areas_classical:
                            captions_area.append(area.lower())
                            image_paths_class_areas.append(img)

                    elif maps_directory == "PictorialMaps":

                        captions_class_pic.append("pictorial map")  # only set map type
                        image_paths_class_pic.append(img)

print(Counter(dates_array))
print("Count- ", len(Counter(dates_array)))

print(Counter(areas_array))
print("Count- ", len(Counter(areas_array)))

print(Counter(notes_array))
print("Count- ", len(Counter(notes_array)))

print(Counter(captions_area))
print("Count- ", len(Counter(captions_area)))

print(Counter(captions_class_pic))
print("Count- ", len(Counter(captions_class_pic)))

# Save lists as numpy arrays for later usage

captions_npy = np.array(captions_area)
np.save("classicalMapsCaptionsArea.npy", captions_npy)

captions_npy = np.array(image_paths_class_areas)
np.save("classicalMapsPathsArea.npy", captions_npy)

captions_npy = np.array(captions_date)
np.save("classicalMapsCaptionsDate.npy", captions_npy)

captions_npy = np.array(image_paths_class_dates)
np.save("classicalMapsPathsDate.npy", captions_npy)

captions_npy = np.array(captions_note)
np.save("classicalMapsCaptionsNote.npy", captions_npy)

captions_npy = np.array(image_paths_class_notes)
np.save("classicalMapsPathsNote.npy", captions_npy)

captions_npy = np.array(captions_class_pic)
np.save("classPictorialCaptions.npy", captions_npy)

captions_npy = np.array(image_paths_class_pic)
np.save("classPictorialPaths.npy", captions_npy)
