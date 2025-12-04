"""A class for keeping track of a particular scintillator's data."""
import copy as copy
import gc as gc
import pandas as pd

import tgfsearch.parameters as params
from tgfsearch.helpers.reader import Reader


class Scintillator:
    """A class used to store data for a single scintillator.

    Parameters
    ----------
    name : str
        The scintillator's name (abbreviated).
    eRC : str
        The scintillator's serial number.

    Attributes
    ----------
    lm_frame : pandas.core.frame.Dataframe
        A pandas dataframe containing all the scintillator's list mode data.
    lm_filelist : list
        A list of list mode files for the day.
    lm_file_ranges : list
        A list of lists. Each sublist contains a pair of numbers corresponding to
        the first and last second in each list mode file.
    lm_file_indices : dict
        A dictionary of lists. Each list contains the indices needed to slice data for a particular file
        out of lm_frame.
    trace_filelist : list
        A list of trace files for the day.
    traces : dict
        A dictionary containing trace data for each of the day's trace files.
    reader : tgfsearch.helpers.reader.Reader
        A Reader object for reading the scintillator's data files.

    """

    def __init__(self, name, eRC):
        self.name = name
        self.eRC = eRC
        self.lm_frame = pd.DataFrame()
        self.lm_filelist = []
        self.lm_file_ranges = []
        self.lm_file_indices = {}
        self.trace_filelist = []
        self.traces = {}
        self.reader = Reader()

    def __del__(self):
        self.clear()

    def __str__(self):
        return f'Scintillator({self.name}, {self.eRC})'

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return self.data_present()

    def data_present(self, data_type='lm'):
        if data_type == 'lm':
            # Using energy as an arbitrary check here. Seconds of day or any of the others would've worked as well
            return True if 'energies' in self.lm_frame and len(self.lm_frame['energies']) > 0 else False
        elif data_type == 'trace':
            return True if len(self.traces) > 0 else False
        else:
            raise ValueError(f"'{data_type}' is not a valid data type.")

    def get_attribute(self, attribute, deepcopy=True):
        """Returns the requested attribute."""
        if hasattr(self, attribute):
            if deepcopy:
                return copy.deepcopy(getattr(self, attribute))
            else:
                return getattr(self, attribute)
        else:
            raise ValueError(f"'{attribute}' is not a valid attribute.")

    def set_attribute(self, attribute, info, deepcopy=True):
        """Updates the requested attribute."""
        if hasattr(self, attribute):
            attribute_type = type(getattr(self, attribute))
            info_type = type(info)
            if info_type == attribute_type:
                if deepcopy:
                    setattr(self, attribute, copy.deepcopy(info))
                else:
                    setattr(self, attribute, info)

            else:
                raise TypeError(f"'{attribute}' must be of type '{attribute_type.__name__}', "
                                f"not '{info_type.__name__}'.")

        else:
            raise ValueError(f"'{attribute}' is not a valid attribute.")

    def get_lm_data(self, column, file_name=None):
        """Returns a single column of list mode data as a numpy array."""
        if file_name is None:
            frame = self.lm_frame
        else:
            frame = self.get_lm_file(file_name, deepcopy=False)

        if column in frame:
            return frame[column].to_numpy()
        else:
            raise ValueError(f"'{column}' is either not a valid column or data hasn't been imported.")

    def set_lm_data(self, column, new_data, file_name=None):
        """Sets a single column of list mode data to the new data specified."""
        if file_name is None:
            frame = self.lm_frame
        else:
            frame = self.get_lm_file(file_name, deepcopy=False)

        if len(frame) != len(new_data):
            raise ValueError(f"length of data ({len(new_data)}) doesn't match size of frame ({len(frame)}).")

        if column in frame:
            frame[column] = new_data
        else:
            raise ValueError(f"'{column}' is either not a valid column or data hasn't been imported.")

    def find_lm_file_index(self, count_time):
        """Returns the index of the list mode file that the given count occurred in."""
        # Checking to see that the count is inside the day or in the ~500 seconds of the next day included sometimes
        if count_time < 0:
            return -1

        # Binary search of list mode file ranges
        low = 0
        high = len(self.lm_file_ranges) - 1
        while low <= high:
            mid = low + (high - low) // 2
            if self.lm_file_ranges[mid][0] <= count_time <= self.lm_file_ranges[mid][1]:
                return mid
            elif self.lm_file_ranges[mid][0] > count_time:
                high = mid - 1
            else:
                low = mid + 1

        return -1

    def find_lm_file(self, count_time):
        """Returns the name of the list mode file that the given count occurred in."""
        index = self.find_lm_file_index(count_time)
        if index != -1:
            return self.lm_filelist[index]
        else:
            return ''

    def get_lm_file(self, file_name, deepcopy=True):
        """Returns the list mode data for the specified list mode file."""
        if file_name in self.lm_file_indices:
            indices = self.lm_file_indices[file_name]
            if deepcopy:
                return self.lm_frame[indices[0]:indices[1]].copy(deep=True)
            else:
                return self.lm_frame[indices[0]:indices[1]]

        else:
            raise ValueError(f"no file '{file_name}' for scintillator '{self.name}'.")

    def get_trace(self, trace_name, deepcopy=True):
        """Returns trace data for the given time id."""
        if trace_name in self.traces:
            if deepcopy:
                return self.traces[trace_name].copy(deep=True)
            else:
                return self.traces[trace_name]

        else:
            raise ValueError(f"No trace with name '{trace_name}' for scintillator '{self.name}'.")

    def get_trace_names(self):
        """Returns a list of names for traces that are currently being stored."""
        return list(self.traces.keys())

    def find_matching_traces(self, count_time, date_str, trace_list=None):
        """Finds the traces that could be a match for the given count (if they exist)."""
        if count_time < 0 or count_time > params.SEC_PER_DAY + 500:  # Checking that count is inside the day
            return []

        if trace_list is None:
            trace_list = self.trace_filelist

        # Get the timestamp of the count in hhmmss format
        timestamp = ''
        carry = int(count_time)
        for num_sec in [params.SEC_PER_HOUR, 60, 1]:
            dial_val = carry // num_sec
            carry = carry % num_sec
            timestamp += '0' + str(dial_val) if dial_val < 10 else str(dial_val)

        # Finding traces that contain the timestamp of the count
        matches = []
        for trace in trace_list:
            # Meant to account for edge cases where the timestamp happens to be the same as the date
            if timestamp == date_str:
                if trace.split('xtr')[1].count(timestamp) >= 2:
                    matches.append(trace)
            else:
                if trace.split('xtr')[1].count(timestamp) >= 1:
                    matches.append(trace)

        return matches

    def clear(self, clear_filelists=True):
        """Clears all data currently stored in the Scintillator."""
        self.lm_frame = pd.DataFrame()
        self.lm_file_ranges.clear()
        self.lm_file_indices.clear()
        self.traces.clear()
        self.reader.reset()
        if clear_filelists:
            self.lm_filelist.clear()
            self.trace_filelist.clear()

        gc.collect()
