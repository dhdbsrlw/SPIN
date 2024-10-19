import os
import braceexpand
import pickle
import tarfile

""" Util Functions """
def load_data(data_dir, recursive=True):
    data_files = list(braceexpand.braceexpand(data_dir))
    if recursive:
        data_files = [os.path.join(root, name)
                            for path in data_files
                            for root, dirs, files in os.walk(path)
                            for name in files if name.endswith('.tar')]
    else:
        data_files = [os.path.join(path, f) for path in data_files for f in os.listdir(path) if f.endswith('.tar')]
    return data_files


def load_data_from_tar(tar_file): # data_dir is a single tar file
    data_list = []
    
    with tarfile.open(tar_file, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.pkl'):
                file = tar.extractfile(member)
                if file is not None:
                    data = pickle.load(file)
                    data_list.append(data)
                    
    return data_list
