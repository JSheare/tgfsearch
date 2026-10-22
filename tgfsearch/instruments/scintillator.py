"""A module containing a class for managing a particular scintillator's data."""
from __future__ import annotations

import copy
import datetime as dt
import gc
import multiprocessing
import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from multiprocessing.connection import Connection
from multiprocessing.pool import Pool
from typing import Dict, List, Tuple

import tgfsearch.config.parameters as params
import tgfsearch.helpers.helper_funcs as helper_funcs
from tgfsearch.tools.reader import Reader


def _read_data_files(reader: Reader, filelist: List[str], clean_energy: bool, connection: Connection) -> None:
    """Reads the given data files and sends them to the given pipe. Meant to be run in a subprocess, which means that
    the passed arguments and piped values are serialized/deserialized on each end."""
    warning_strings = []
    with warnings.catch_warnings():
        # Redirecting warning strings to the warning_strings list
        warnings.showwarning = (lambda message, category, filename, lineno,
                                       file_handle=None, line=None: warning_strings.append(
            warnings.formatwarning(message, category, filename, lineno, line)))
        for file in filelist:
            try:
                data = reader.read(file, clean_energy=clean_energy)
                connection.send(data)
                # Sending the warning strings, if they exist
                if len(warning_strings) > 0:
                    connection.recv()
                    connection.send(warning_strings)
                    warning_strings.clear()

            except Exception as ex:
                connection.send(ex)

            connection.recv()

    connection.send(reader)  # Sending the updated reader back to the main process


class Scintillator:
    """A class used to store data for a single scintillator.

    Parameters
    ----------
    name : str
        The scintillator's name (abbreviated).
    erc : str
        The scintillator's serial number.
    lm_format : str
        The format of the scintillator's list mode files.
    date : datetime.date
        The date that the Scintillator holds data for. If the Scintillator holds data for multiple dates, it will be the
        earliest one.

    Attributes
    ----------
    dates_stored : list of datetime.date
        A list of dates currently being stored in the Instrument.
    _lm_frame : pandas.core.frame.Dataframe
        A pandas dataframe containing all the scintillator's list mode data.
    _lm_filelist : list of str
        A list of list mode files for the day.
    _lm_file_ranges : list of list of float
        A list of lists. Each sublist contains a pair of numbers corresponding to
        the first and last second in each list mode file.
    _lm_file_indices : dict of [str, list of int]
        A dictionary of lists. Each list contains the indices needed to slice data for a particular file
        out of lm_frame.
    _trace_filelist : list of str
        A list of trace files for the day.
    _traces : dict of [str, pandas.core.frame.DataFrame]
        A dictionary containing trace data for each of the day's trace files.
    _reader : tgfsearch.tools.reader.Reader
        A Reader object for reading the scintillator's data files.

    """

    def __init__(self, name: str, erc: str, date: dt.date, lm_format: str) -> None:
        self.name = name
        self.erc = erc
        self.date = date
        self.lm_format = lm_format
        self.dates_stored = []

        self._lm_frame = pd.DataFrame()
        self._lm_filelist = []
        self._lm_file_ranges = []
        self._lm_file_indices = {}

        self._trace_filelist = []
        self._traces = {}

        self._reader = Reader()

    def __del__(self) -> None:
        self.clear()

    def __str__(self) -> str:
        return f'Scintillator({self.name}, {self.erc})'

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return self.data_present()

    def data_present(self, data_type: str = 'lm') -> bool:
        """Returns True if the specified data has been imported, and False otherwise."""
        if data_type == 'lm':
            # Using energy as an arbitrary check here. Seconds of day or any of the others would've worked as well
            return True if 'energies' in self._lm_frame and len(self._lm_frame['energies']) > 0 else False
        elif data_type == 'trace':
            return True if len(self._traces) > 0 else False
        else:
            raise ValueError(f"'{data_type}' is not a valid data type.")

    def get_identity(self) -> Dict[str, str | dt.date]:
        """Returns a dictionary containing the scintillator's identifying information."""
        return {'name': self.name, 'erc': self.erc, 'date': copy.deepcopy(self.date), 'lm_format': self.lm_format,
                'dates_stored': copy.deepcopy(self.dates_stored)}

    def get_reader(self) -> Reader:
        """Returns the scintillator's Reader object."""
        return self._reader

    def set_reader(self, reader: Reader) -> None:
        """Sets the scintillator's Reader object to the given one."""
        self._reader = reader

    def get_lm_filelist(self, deepcopy: bool = True) -> List[str]:
        """Returns either the list of list mode files or a copy depending on the optional deepcopy param."""
        if deepcopy:
            return copy.deepcopy(self._lm_filelist)

        return self._lm_filelist

    def set_lm_filelist(self, lm_filelist, deepcopy: bool = True) -> None:
        """Sets either the given list of list mode files or a copy depending on the optional deepcopy param."""
        if deepcopy:
            self._lm_filelist = copy.deepcopy(lm_filelist)
        else:
            self._lm_filelist = lm_filelist

    def get_lm_data(self, column: str, file_name: str | None = None) -> npt.NDArray[np.float64]:
        """Returns a single column of list mode data as a numpy array."""
        if file_name is None:
            frame = self._lm_frame
        else:
            frame = self.get_lm_file(file_name, deepcopy=False)

        if column in frame.columns:
            return frame[column].to_numpy()
        else:
            raise ValueError(f"'{column}' is either not a valid column or data hasn't been imported.")

    def set_lm_data(self, column: str, new_data: npt.NDArray[np.float64], file_name: str | None = None) -> None:
        """Sets a single column of list mode data to the new data specified."""
        if file_name is None:
            frame = self._lm_frame
        else:
            frame = self.get_lm_file(file_name, deepcopy=False)

        if len(frame) != len(new_data):
            raise ValueError(f"length of data ({len(new_data)}) doesn't match size of frame ({len(frame)}).")

        if column in frame.columns:
            frame[column] = new_data
        else:
            raise ValueError(f"'{column}' is either not a valid column or data hasn't been imported.")

    def get_lm_frame(self, deepcopy: bool = True) -> pd.DataFrame:
        """Returns either the original list mode data frame or a copy depending on the optional deepcopy param."""
        if deepcopy:
            return self._lm_frame.copy(deep=True)

        return self._lm_frame

    def set_lm_frame(self, lm_frame: pd.DataFrame, deepcopy: bool = True) -> None:
        """Sets either the given list mode data frame or a copy depending on the optional deepcopy param."""
        if deepcopy:
            self._lm_frame = lm_frame.copy(deep=True)
        else:
            self._lm_frame = lm_frame

    def _find_lm_file_index(self, count_time: float) -> int:
        """Returns the index of the list mode file that the given count occurred in."""
        # Checking to see that the count is inside the day or in the ~500 seconds of the next day included sometimes
        if count_time < 0:
            return -1

        # Binary search of list mode file ranges
        low = 0
        high = len(self._lm_file_ranges) - 1
        while low <= high:
            mid = low + (high - low) // 2
            if self._lm_file_ranges[mid][0] <= count_time <= self._lm_file_ranges[mid][1]:
                return mid
            elif self._lm_file_ranges[mid][0] > count_time:
                high = mid - 1
            else:
                low = mid + 1

        return -1

    def find_lm_file(self, count_time: float) -> str:
        """Returns the name of the list mode file that the given count occurred in."""
        index = self._find_lm_file_index(count_time)
        if index != -1:
            return self._lm_filelist[index]
        else:
            return ''

    def get_lm_file(self, file_name: str, deepcopy: bool = True) -> pd.DataFrame:
        """Returns the list mode data for the specified list mode file."""
        if file_name in self._lm_file_indices:
            indices = self._lm_file_indices[file_name]
            if deepcopy:
                return self._lm_frame[indices[0]:indices[1]].copy(deep=True)
            else:
                return self._lm_frame[indices[0]:indices[1]]

        else:
            raise ValueError(f"no file '{file_name}' for scintillator '{self.name}'.")

    def get_trace_filelist(self, deepcopy: bool = True) -> List[str]:
        """Returns either the list of list mode files or a copy depending on the optional deepcopy param."""
        if deepcopy:
            return copy.deepcopy(self._trace_filelist)

        return self._trace_filelist

    def set_trace_filelist(self, trace_filelist, deepcopy: bool = True) -> None:
        """Sets either the given list of trace files or a copy depending on the optional deepcopy param."""
        if deepcopy:
            self._trace_filelist = copy.deepcopy(trace_filelist)
        else:
            self._trace_filelist = trace_filelist

    def get_trace_names(self) -> List[str]:
        """Returns a list of names for traces that are currently being stored."""
        return list(self._traces.keys())

    def get_trace(self, trace_name: str, deepcopy: bool = True) -> pd.DataFrame:
        """Returns trace data for the given time id."""
        if trace_name in self._traces:
            if deepcopy:
                return self._traces[trace_name].copy(deep=True)
            else:
                return self._traces[trace_name]

        else:
            raise ValueError(f"No trace with name '{trace_name}' for scintillator '{self.name}'.")

    def find_matching_traces(self, count_time: float, date_str: str, trace_list: List[str] | None = None) -> List[str]:
        """Finds the traces that could be a match for the given count (if they exist)."""
        if count_time < 0 or count_time > params.SEC_PER_DAY + 500:  # Checking that count is inside the day
            return []

        if trace_list is None:
            trace_list = self._trace_filelist

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

    def clear(self, clear_filelists: bool = True) -> None:
        """Clears all data currently stored in the Scintillator."""
        self._lm_frame = pd.DataFrame()
        self._lm_file_ranges.clear()
        self._lm_file_indices.clear()
        self._traces.clear()
        self._reader.reset()
        if clear_filelists:
            self._lm_filelist.clear()
            self._trace_filelist.clear()
            self.dates_stored = [self.date]

        gc.collect()

    def get_projected_memory(self, include_lm: bool = True, include_traces: bool = True) -> float:
        """Returns the projected size (in bytes) of the object after all currently listed files have been imported."""
        try:
            total_file_size = 0
            if include_lm:
                for file in self._lm_filelist:
                    total_file_size += helper_funcs.file_size(file) * params.LM_GROWTH_FACTORS[self.lm_format]

            if include_traces:
                for file in self._trace_filelist:
                    total_file_size += helper_funcs.file_size(file) * params.TRACE_GROWTH_FACTORS[self.lm_format]

            return total_file_size
        except KeyError:
            raise KeyError(f"growth factor for '{self.lm_format}' missing.")

    def assemble_filelists(self, import_loc: str) -> None:
        """Assembles the lists of data files for this scintillator in the given import location and stores them."""
        lm_extensions = ['.csv', '.txt']
        trace_extensions = ['.xtr']

        complete_filelist = helper_funcs.get_serial_num_filelist(import_loc, self.erc, self.date)
        unique_files = set()
        lm_filelist = []
        trace_filelist = []
        for file in complete_filelist:
            if len(file) >= 4:
                dot_index = len(file) - 4 if file[-3:] == '.gz' else len(file) - 1
                end = dot_index + 1
                while file[dot_index] != '.' and dot_index >= 0:
                    dot_index -= 1

                extension = file[dot_index:end]
                if ((extension in lm_extensions or extension in trace_extensions) and
                        file[:dot_index] not in unique_files):
                    if extension in trace_extensions:
                        trace_filelist.append(file)
                    else:
                        lm_filelist.append(file)

        lm_filelist.sort()
        trace_filelist.sort()

        self._lm_filelist = lm_filelist
        self._trace_filelist = trace_filelist

    def import_lm(self, process_pool: Pool, options: Dict[str, bool]) -> Tuple[int, str]:
        """Imports list mode data."""
        if len(self.dates_stored) > 1:
            raise RuntimeError("cannot import multiple days' data.")

        file_frames = []
        file_ranges = []
        file_indices = {}
        start_index = 0
        output_strings = ['List Mode Files:\nFile|Import Success|\n']

        # Importing the data
        end1, end2 = multiprocessing.Pipe()
        file_index = 0
        process_pool.apply_async(_read_data_files,
                                 args=(self._reader, self._lm_filelist, options['clean_energy'], end2))
        while True:
            data = end1.recv()  # Reading data from the other process
            end1.send(1)  # Notifying the other process that the data has been received
            if isinstance(data, pd.DataFrame):
                pass
            elif isinstance(data, list):
                # Outputting any warnings from the previous file
                output_strings.append('\t' + '\t'.join(data))
                continue
            elif isinstance(data, Reader):
                # Storing the updated reader object from the other process, which also serves as a sentinel value
                self._reader = data
                break
            else:
                # Outputting any reader errors
                output_strings.append(f'{self._lm_filelist[file_index]}|False|\n\tError importing file: {data}\n')
                file_index += 1
                continue

            first_second = data['SecondsOfDay'].min()
            last_second = data['SecondsOfDay'].max()

            # Determines the time gaps between adjacent files
            file_ranges.append([first_second, last_second])
            file_frames.append(data)
            output_strings.append(f'{self._lm_filelist[file_index]}|True|\n')

            # Keeps track of file indices in the larger dataframe
            data_length = len(data.index)
            file_indices[self._lm_filelist[file_index]] = [start_index, start_index + data_length]
            start_index += data_length

            file_index += 1

        if len(file_frames) > 0:
            # Correcting for the fact that the first few minutes of the next day are usually included in the last file
            # First count of the next day - last count of the current day will be either equal to or less than
            # -params.SEC_PER_DAY by up to a few hundred seconds depending on how late the first count came in.
            # Choosing a large error just to be sure
            error = 500
            day_change = np.where(file_frames[-1]['SecondsOfDay'].diff() <= -(params.SEC_PER_DAY - error))[0]
            if len(day_change) > 0:
                file_frames[-1].iloc[
                    day_change[0]:, file_frames[-1].columns.get_loc('SecondsOfDay')] += params.SEC_PER_DAY

            # Correcting the last file's time ranges too
            if file_ranges[-1][1] - file_ranges[-1][0] <= -(params.SEC_PER_DAY - error):
                file_ranges[-1][1] += params.SEC_PER_DAY

            # Makes the final dataframe and stores it
            lm_frame = pd.concat(file_frames, ignore_index=True)
            if 'index' in lm_frame.columns:
                lm_frame.drop(columns='index', inplace=True)

            self._lm_frame = lm_frame
            self._lm_file_ranges = file_ranges
            self._lm_file_indices = file_indices

            output_strings.append(f'\nTotal Counts: {len(self._lm_frame.index)}\n\n')

        else:
            output_strings.append('\n')

        return len(file_frames), ''.join(output_strings)

    def import_traces(self, process_pool: Pool, options: Dict[str, bool]) -> Tuple[int, str]:
        """Imports trace data."""
        if len(self.dates_stored) > 1:
            raise RuntimeError("cannot import multiple days' data.")

        traces = {}
        output_strings = ['Trace Files:\nFile|Import Success|\n']

        # Importing the data
        end1, end2 = multiprocessing.Pipe()
        file_index = 0
        process_pool.apply_async(_read_data_files, args=(self._reader, self._trace_filelist,
                                                        options['clean_energy'], end2))
        while True:
            data = end1.recv()  # Reading data from the other process
            end1.send(1)  # Notifying the other process that the data has been received
            if isinstance(data, pd.DataFrame):
                pass
            elif isinstance(data, list):
                # Outputting any warnings from the previous file
                output_strings.append('\t' + '\t'.join(data))
                continue
            elif isinstance(data, Reader):
                # Storing the updated reader object from the other process, which also serves as a sentinel value
                self._reader = data
                break
            else:
                # Outputting any reader errors
                output_strings.append(f'{self._trace_filelist[file_index]}|False|\n\tError importing file: {data}\n')
                file_index += 1
                continue

            traces[self._trace_filelist[file_index]] = data
            output_strings.append(f'{self._trace_filelist[file_index]}|True|\n')

            file_index += 1

        # Storing the traces
        if len(traces) > 0:
            self._traces = traces

        output_strings.append('\n')

        return len(traces), ''.join(output_strings)

    def splice(self, operand_scintillator: Scintillator) -> Scintillator:
        """Returns a new Scintillator with the combined data of the current Scintillator and the one provided."""
        if operand_scintillator.name == self.name:
            if self.date < operand_scintillator.date:
                earlier = self
                later = operand_scintillator
            elif self.date > operand_scintillator.date:
                earlier = operand_scintillator
                later = self
            else:
                raise ValueError('cannot splice the same day into itself.')

            new_erc = earlier.erc if earlier.erc == later.erc else f'{earlier.erc}/{later.erc}'
            new_lm_format = earlier.lm_format if earlier.lm_format == later.lm_format else \
                f'{earlier.lm_format}/{later.lm_format}'
            new_scintillator = type(earlier)(earlier.name, new_erc, copy.deepcopy(earlier.date), new_lm_format)
            new_scintillator.dates_stored.append(copy.deepcopy(later.date))

            day_difference = (later.date - earlier.date).days

            # Combining list mode file lists
            new_scintillator._lm_filelist = earlier._lm_filelist + later._lm_filelist

            # Combining list mode file ranges
            # Updating ranges from later Scintillator to reflect the date difference
            new_ranges = copy.deepcopy(later._lm_file_ranges)
            for new_range in new_ranges:
                new_range[0] += day_difference * params.SEC_PER_DAY
                new_range[1] += day_difference * params.SEC_PER_DAY

            new_scintillator._lm_file_ranges = copy.deepcopy(earlier._lm_file_ranges) + new_ranges

            # Combining list mode file indices
            # Updating indices from later Scintillator to reflect their new position in the data frame
            new_start = len(earlier._lm_frame)
            new_indices = copy.deepcopy(later._lm_file_indices)
            for file in new_indices:
                new_indices[file][0] += new_start
                new_indices[file][1] += new_start

            new_indices.update(copy.deepcopy(earlier._lm_file_indices))

            # Combining list mode data frames
            # Updating later frame to reflect the difference in days
            later_frame = later._lm_frame.copy(deep=True)
            later_frame['SecondsOfDay'] += day_difference * params.SEC_PER_DAY
            new_scintillator._lm_frame = pd.concat([earlier._lm_frame, later_frame], axis=0)

            # Combining trace file lists
            new_scintillator._trace_filelist = earlier._trace_filelist + later._trace_filelist

            # Combining trace tables
            for trace in earlier._traces:
                new_scintillator._traces[trace] = earlier._traces[trace].copy(deep=True)

            for trace in later._traces:
                new_scintillator._traces[trace] = later._traces[trace].copy(deep=True)

            return new_scintillator

        else:
            raise TypeError(f"cannot splice '{self.name}' with '{operand_scintillator.name}'.")
