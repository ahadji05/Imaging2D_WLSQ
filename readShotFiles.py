
import glob
import re

def returnShotIndices(path_to_shots_dir, file_extension, low_bound_str, up_bound_str):
    filenames = glob.glob(str(path_to_shots_dir+"*."+file_extension))

    #find shot indices from the filenames
    shot_isx = []
    for i in range(len(filenames)):
        xloc = re.search(path_to_shots_dir+low_bound_str+"(.+?)"+up_bound_str, filenames[i])
        shot_isx.append(int(xloc.group(1))) #return index as int

    return shot_isx, filenames #return indices and files-names
    