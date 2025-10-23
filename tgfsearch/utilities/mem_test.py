"""A script that measures the average bytes of Detector memory added for every byte of data file imported."""
import gc as gc
import numpy as np
import os as os
import pandas as pd
import sys as sys

# Adds grandparent directory to sys.path. Necessary to make the imports below work when running this file as a script
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import tgfsearch.tools as tl
from tgfsearch.search import is_valid_search, get_detector

def get_obj_size(obj, visited=None):
    if isinstance(obj, pd.DataFrame):
        size = obj.memory_usage(deep=True).sum()
    elif isinstance(obj, np.ndarray):
        size = obj.nbytes
    else:
        size = sys.getsizeof(obj)

    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return 0

    visited.add(obj_id)
    if isinstance(obj, pd.DataFrame):
        pass
    elif isinstance(obj, dict):
        for key in obj.keys():
            size += get_obj_size(key, visited)

        for value in obj.values():
            size += get_obj_size(value, visited)
    elif hasattr(obj, '__dict__'):
        # Iterate through all the object's data members
        size += get_obj_size(obj.__dict__, visited)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            size += get_obj_size(item, visited)

    return size


def main():
    if len(sys.argv) >= 4:
        first_date = str(sys.argv[1])
        second_date = str(sys.argv[2])
        unit = str(sys.argv[3])
    else:
        print('Error: please provide a first date, a second date, and a unit name.')
        exit()

    if len(sys.argv) > 4:
        mode_info = sys.argv[4:]
    else:
        mode_info = []

    # Flag for a custom import directory
    if len(mode_info) >= 2 and '-c' in mode_info:
        import_loc = mode_info[mode_info.index('-c') + 1]
    else:
        import_loc = ''

    # Flag for ignoring scintillators
    if len(mode_info) >= 2 and '-i' in mode_info:
        index = mode_info.index('-i')
        if index < len(mode_info) - 1:
            end_index = index + 1
            for i in range(index + 1, len(mode_info)):
                if '-' in mode_info[i]:
                    break

                end_index += 1

            ignore_scints = mode_info[index + 1:end_index]
        else:
            ignore_scints = []

    else:
        ignore_scints = []

    # Makes sure inputs are valid
    if not is_valid_search(first_date, second_date, unit, print_feedback=True):
        exit()

    lm_growth_factors = []
    trace_growth_factors = []
    try:
        for date_str in tl.make_date_list(first_date, second_date):
            print(f'Importing data for {date_str}...')
            detector = get_detector(unit, date_str)
            if import_loc != '':
                try:
                    detector.set_import_loc(import_loc)
                    if not detector.has_identity():
                        raise FileNotFoundError("couldn't infer identity.")

                except FileNotFoundError:
                    print('Error: no data files to infer detector identity from. Please provide the import location of '
                          'the data.')
                    exit()

            import_scints = []
            for scintillator in detector:
                if scintillator not in ignore_scints:
                    import_scints.append(scintillator)

            # Measuring the growth factor for the list mode files
            detector.import_data(import_traces=False, import_scints=import_scints, feedback=True)
            fileset_size = 0
            for scintillator in detector:
                lm_filelist = detector.get_attribute(scintillator, 'lm_filelist', deepcopy=False)
                for file in lm_filelist:
                    fileset_size += tl.file_size(file)

            if fileset_size > 0:
                lm_growth_factors.append(get_obj_size(detector) / fileset_size)

            detector.clear()
            gc.collect()

            # Measuring the growth factor for the trace files
            detector.import_data(import_lm=False, import_scints=import_scints, feedback=True)
            fileset_size = 0
            for scintillator in detector:
                trace_filelist = detector.get_attribute(scintillator, 'trace_filelist', deepcopy=False)
                for file in trace_filelist:
                    fileset_size += tl.file_size(file)

            if fileset_size > 0:
                trace_growth_factors.append(get_obj_size(detector) / fileset_size)

            detector.clear()
            gc.collect()
            print('\n')

        # Printing the final average growth factors
        if len(lm_growth_factors) > 0:
            print(f'Average list mode data growth factor: {sum(lm_growth_factors) / len(lm_growth_factors)}.')
        else:
            print('No list mode data growth factors to average.')

        if len(trace_growth_factors) > 0:
            print(f'Average trace data growth factor: {sum(trace_growth_factors) / len(trace_growth_factors)}.')
        else:
            print('no trace data growth factors to average.')

        print('\n')

    except MemoryError:
        print('Error: not enough memory available on system to complete the test.')
        exit()


if __name__ == '__main__':
    main()
