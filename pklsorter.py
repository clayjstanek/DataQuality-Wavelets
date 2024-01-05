from pathlib import Path
import shutil

p = Path(__file__).parents[0]
data_path = p.joinpath("dqs", "coherenceraw090323")
pkl_folders = ["pkl_black", "pkl_white", "pkl_brain", "pkl_oval", "pkl_red", "pkl_spikes"]
pics_folders = ["pics_black", "pics_white", "pics_brain", "pics_oval", "pics_red", "pics_spikes"]

# Check for pkl destination folders and create if needed
for i in pkl_folders:
    path = p.joinpath(i)
    if not path.exists():
        Path.mkdir(path, parents=True)
        print("Created folder: " + i)
    else:
        print("Pkl folder found: " + str(path).split('/')[-1])

if data_path.exists():
    print("Raw data folder found: " + str(data_path).split('/')[-1])
    # Iterate through all pics directories
    for i in range(len(pics_folders)):
        pics_path = p.joinpath(pics_folders[i])
        dest_path = p.joinpath(pkl_folders[i])
        if pics_path.exists():
            print("Images folder found: " + str(pics_path).split('/')[-1])
            print("Copying pkl files...")
            # Iterate through all images in current pics directory
            for i in pics_path.glob("*.png"):
                imgname = str(i).split('/')[-1]
#                newpath = str(data_path) + '/' + imgname.replace("coherence_pics", "wavecoherence_data").replace(".png", ".pkl")
#                newpath = imgname.replace("coherence_pics", "wavecoherence_data").replace(".png", ".pkl")
                newpath = data_path.joinpath(Path((i.stem).replace("coherence_pics", "wavecoherence_data")).with_suffix( ".pkl"))
                if Path(newpath).exists():
                    shutil.copy(newpath, dest_path)
                else:
                    #print("Pkl not found: " + str(newpath).split('/')[-1]) # Enable to display missing pkl files
                    continue
        else:
            raise Exception("Folder not found: " + str(pics_path).split('/')[-1])
else:
    raise Exception("Folder not found: " + str(data_path).split('/')[-1])
