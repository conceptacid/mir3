import os, sys

##################################################################################################

def collect_filename(audio_path, name, files_by_instrument):
    dir = os.path.join(audio_path, name)
    yaml_file = name + '_METADATA.yaml'
    path = os.path.join(dir, yaml_file)

    stream = open(path, "r")
    doc = yaml.load(stream)
    raw_dir =  name + '_RAW'

    for k,s in doc['stems'].items():
        for k, r in s['raw'].items():
            instrument = group_by_instrument.get(  r['instrument']  )
            if instrument == None:
                continue
            filename = os.path.join(name, raw_dir, r['filename'])
            if not files_by_instrument.has_key(instrument):
                files_by_instrument[instrument] = []
            files_by_instrument[instrument].append(filename)

##################################################################################################            

def collect_all_filenames(audio_path):
    files_by_instrument = {}
    dirs = [dir for dir in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path,dir))]
    for name in dirs:
        collect_filename(audio_path, name, files_by_instrument)
    return files_by_instrument

##################################################################################################    