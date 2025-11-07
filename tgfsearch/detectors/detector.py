"""A base class for keeping track of lightning data and associated information."""
import datetime as dt
import gc as gc
import glob as glob
import json as json
import multiprocessing as multiprocessing
import numpy as np
import os as os
import pandas as pd
import psutil as psutil
import sys as sys
import threading as threading
import warnings as warnings

import tgfsearch.parameters as params
import tgfsearch.tools as tl
from tgfsearch.detectors.scintillator import Scintillator
from tgfsearch.helpers.reader import Reader


class Detector:
    """A class used to store all relevant information about an instrument and its data for a day.

    The Detector class is used to store the name of the detector, the date in various formats,
    and the actual data for the requested day in a single, centralized location.

    Parameters
    ----------
    unit : str
        The name of the instrument.
    date_str : str
        The timestamp for the requested day in yymmdd format.

    Attributes
    ----------
    log : _io.TextIO
        The file where actions and findings are logged.
    first_sec : float
        The first second of the day in EPOCH time.
    full_date_str : str
        The timestamp for the requested in day in yyyy-mm-dd format.
    dates_stored : list
        A list of dates currently being stored in the Detector.
    _has_identity : bool
        A flag for whether the Detector has an identity (established name, scintillator configuration, etc.).
    lm_growth_factors : dict
        For each scintillator, the average bytes of Detector memory added per byte of list mode data file imported.
    trace_growth_factors : dict
        For each scintillator, the average bytes of Detector memory added per byte of trace mode data file imported.
    _import_loc : str
        The directory where data files for the day are located.
    _results_loc : str
        The directory where results will be exported.
    _scintillators : dict
        A dictionary containing Scintillators. These keep track of data for each of the instrument's
        scintillators. Note the name mangling underscore.
    scint_list : list
        A list of the instrument's scintillator names.
    default_scintillator : str
        A string representing the default scintillator.
    deployment : dict
        Deployment information for the instrument on the requested day (if available).

    """

    def __init__(self, unit, date_str, **kwargs):
        # Basic information
        self.date_str = date_str  # yymmdd
        self.log = None
        self.first_sec = tl.get_first_sec(self.date_str)
        self.full_date_str = dt.datetime.fromtimestamp(int(self.first_sec), dt.UTC).strftime('%Y-%m-%d')  # yyyy-mm-dd
        self.dates_stored = [date_str]

        # Identity-related information
        self._has_identity = False
        self.unit = unit.upper()
        self.lm_growth_factors = {}
        self.trace_growth_factors = {}
        self._import_loc = ''
        self._results_loc = ''
        self._scintillators = {}
        self.scint_list = []
        self.default_scintillator = ''
        self.deployment = self._get_deployment()

        # Allows us to disable identity reading if we need to, but without advertising it in the documentation
        if 'read_identity' in kwargs and not kwargs['read_identity']:
            pass
        else:
            self._read_identity()

        self.set_results_loc(os.getcwd().replace('\\', '/'))

    def __del__(self):
        self.clear()

    def __str__(self):
        """String casting overload. Returns a string of the form 'Detector(unit, date_str)'."""
        return f'Detector({self.unit}, {self.date_str})'

    # Debugging string dunder
    def __repr__(self):
        """Debugging string dunder method. Returns a string of the form 'Detector(unit, date_str)' along
        with some info about which Scintillators have data."""
        scintillators_with_data = []
        has_data = False
        for scintillator in self._scintillators:
            if self._scintillators[scintillator]:
                has_data = True
                scintillators_with_data.append(scintillator)

        default_string = self.__str__()
        data_string = f' in {scintillators_with_data}' if has_data else ''
        return default_string + f' Has data = {has_data} ' + data_string

    def __iter__(self):
        """Iterator dunder. Returns a generator that yields the Detector's scintillator names."""
        for scintillator in self.scint_list:
            yield scintillator

    def __bool__(self):
        """Bool casting overload. Returns True if data for the default scintillator is present."""
        if self._has_identity:
            return self.data_present_in(self.default_scintillator)

        return False

    def __contains__(self, scintillator):
        """Contains overload. Returns True if the provided string corresponds to a scintillator in Detector,
        False otherwise."""
        return scintillator in self._scintillators

    def __add__(self, operand_detector):
        """Addition operator overload. Returns a new Detector containing data from the current Detector
        and the provided one. See splice() method documentation for more information."""
        return self.splice(operand_detector)

    def _get_deployment(self):
        """Returns a dictionary full of deployment information for the instrument on its specified date."""
        for file in glob.glob(
                f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/deployments/'
                f'{self.unit}_deployment_*.json'):
            file_dates = file.split('deployment')[-1][1:].split('.')[0].split('_')
            if int(file_dates[0]) <= int(self.date_str) <= int(file_dates[1]):
                with open(file, 'r') as deployment:
                    return json.load(deployment)

        return {'location': 'no location listed', 'instrument': self.unit, 'start_date': '000000', 'end_date': '000000',
                'utc_to_local': 0.0, 'dst_in_region': False, 'weather_station': '', 'sounding_station': '',
                'latitude': 0., 'longitude': 0., 'altitude': 0.,
                'notes': ''}

    def _read_identity(self):
        """Gets and fills in the identity of the Detector from the config file based on the name and date."""
        try:
            with open(f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config/detector_config.json',
                      'r') as file:
                entries = json.load(file)
        except json.decoder.JSONDecodeError:
            raise SyntaxError('invalid syntax in detector config file.')

        if self.unit in entries['identities']:
            identity = entries['identities'][self.unit]
            try:
                self._import_loc = f'{entries["default_data_root"]}/{identity["subtree"]}/{self.date_str}'

                # Getting the right scintillator configuration based on the date
                correct_date_str = ''
                for after_date_str in identity['scintillators']:
                    if int(self.date_str) >= int(after_date_str):
                        correct_date_str = after_date_str
                    else:
                        break

                for scintillator in identity['scintillators'][correct_date_str]:
                    scint_entry = identity['scintillators'][correct_date_str][scintillator]
                    if scintillator == 'default':
                        self.default_scintillator = scint_entry
                    else:
                        self._scintillators[scintillator] = Scintillator(scintillator, scint_entry['eRC'])
                        self.lm_growth_factors[scintillator] = (
                            entries['growth_factors'][scint_entry['file_format']]['lm_growth_factor'])
                        self.trace_growth_factors[scintillator] = (
                            entries['growth_factors'][scint_entry['file_format']]['trace_growth_factor'])
                        self.scint_list.append(scintillator)

                self._has_identity = True
            except KeyError:
                raise SyntaxError(f'missing or incomplete information in detector config file for {self.unit}.')

        else:
            raise ValueError(f"'no entry for {self.unit}' in detector config file.")

    def has_identity(self):
        """Returns True if the Detector has an established identity (established name, scintillator configuration,
        etc.), False otherwise."""
        return self._has_identity

    def file_form(self, eRC):
        """Returns the regex for a scintillator's files given the scintillator's eRC serial number."""
        return f'eRC{eRC}*_*_{self.date_str}_*'

    def get_import_loc(self):
        """Returns the directory where data will be imported from.

        Returns
        -------
        str
            The import directory as a string.

        """

        return self._import_loc

    def set_import_loc(self, loc):
        """Sets the directory where data will be imported from.

        Parameters
        ----------
        loc : str
            The import directory as a string.

        """

        loc.replace('\\', '/')
        if len(loc) > 0:
            if loc[-1] == '/':
                loc = loc[:-1]

        self._import_loc = loc

    def get_results_loc(self):
        """Returns the directory where all results will be stored.

        Returns
        -------
        str
            The results directory as a string.

        """

        return self._results_loc

    def set_results_loc(self, loc, subdir=True):
        """Sets the directory where all results will be stored.

        Parameters
        ----------
        loc : str
            The results directory as a string.

        subdir : bool
            Optional. If True, results will be exported to a subdirectory of the form 'Results/unit/yymmdd' inside
            the specified location rather than straight to the location itself. True by default.

        """

        loc.replace('\\', '/')
        if len(loc) > 0:
            if loc[-1] == '/':
                loc = loc[:-1]

        if subdir:
            self._results_loc = loc + f'/Results/{self.unit}/{self.date_str}'
        else:
            self._results_loc = loc

    def is_named(self, name):
        """Returns True if the Detector has the same name as the passed string.

        Parameters
        ----------
        name : str
            The name being queried.

        Returns
        -------
        bool
            True if the name matches the name of the Detector, False otherwise.

        """

        return name.upper() in self.unit

    def is_valid_scintillator(self, scintillator):
        """Returns True if the Detector contains the specified scintillator, False otherwise

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.

        Returns
        -------
        bool
            True if the scintillator is in Detector, False otherwise.

        """

        return scintillator in self._scintillators

    def data_present_in(self, scintillator, data_type='lm'):
        """Returns True if data is present for the specified scintillator and False otherwise.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.
        data_type : str
            Optional. The type of data to check for. Use 'lm' to check for list mode data and 'trace' to check
            for trace data. Checks for list mode data by default.

        Returns
        -------
        bool
            True if data is present in the specified scintillator, False otherwise.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].data_present(data_type)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_attribute(self, scintillator, attribute, deepcopy=True):
        """Returns the requested attribute for a particular scintillator.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.
        attribute : str
            The name of the attribute of interest.
        deepcopy : bool
            Optional. If True, a deepcopy of the requested attribute will be returned. True by default.

        Returns
        -------
        str || list || numpy.ndarray || dict || pandas.core.frame.DataFrame
            String if 'eRC' is requested; list if 'lm_filelist' or 'lm_file_ranges' is requested;
            numpy array  if 'time' or 'energies' is requested; Reader if 'reader' is requested; dataframe
            if 'lm_frame' is requested, etc.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_attribute(attribute, deepcopy)

        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def set_attribute(self, scintillator, attribute, new_info, deepcopy=True):
        """Updates the requested attribute for a particular scintillator.
        Note: new info must be of the same type as the old.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.
        attribute : str
            The name of the attribute of interest.
        new_info : any
            The new information for the requested attribute.
        deepcopy : bool
            Optional. If True, the requested attribute will be set to a deepcopy of new_info. True by default.

        """

        if scintillator in self._scintillators:
            self._scintillators[scintillator].set_attribute(attribute, new_info, deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_lm_data(self, scintillator, column, file_name=None):
        """Returns a single column of list mode data as a numpy array.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.
        column : str
            The column of interest.
        file_name : str
            Optional. The name of the file to get data for. If not specified,
            data will be retrieved for the whole day.

        Returns
        -------
        numpy.ndarray
            A numpy array containing the requested data column.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_lm_data(column, file_name)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def set_lm_data(self, scintillator, column, new_data, file_name=None):
        """Sets a single column of list mode data to the new data specified.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.
        column : str
            The column of interest.
        new_data : numpy.ndarray
            A numpy array containing the new data.
        file_name : str
            Optional. The name of the file to set data for. If not specified,
            data will be set for the whole day.

        """

        if scintillator in self._scintillators:
            self._scintillators[scintillator].set_lm_data(column, new_data, file_name)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def find_lm_file_index(self, scintillator, count_time):
        """Helper function. Returns the index of the list mode file that the given count occurred in.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator that the count is from.
        count_time : float
            The time that the count occurred at (in seconds of day).

        Returns
        -------
        int
            The index of the list mode file that the given count occurred in. Returns -1 if the count
            isn't in any of the files.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].find_lm_file_index(count_time)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def find_lm_file(self, scintillator, count_time):
        """Returns the name of the list mode file that the given count occurred in.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator that the count is from.
        count_time : float
            The time that the count occurred at (in seconds of day).

        Returns
        -------
        str
            The name of the list mode file that the given count occurred in. Returns an empty string if the count
            isn't in any of the files.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].find_lm_file(count_time)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_lm_file(self, scintillator, file_name, deepcopy=True):
        """Returns the list mode data for the specified list mode file.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator that the file is from.
        file_name : str
            The name of the file that data is being requested for. Note that this must be the *full* name of the file,
            including the path from the root directory.
        deepcopy : bool
            Optional. If True, a deepcopy of the file frame will be returned. True by default.

        Returns
        -------
        pandas.core.frame.DataFrame
            A dataframe with the data for the requested list mode file.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_lm_file(file_name, deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_trace(self, scintillator, trace_name, deepcopy=True):
        """Returns the trace data for the given scintillator and trace name.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.
        trace_name : str
            The name of the trace file.
        deepcopy : bool
            Optional. If True, a deepcopy of the trace frame will be returned. True by default.

        Returns
        -------
        pandas.core.frame.DataFrame
            A dataframe containing the trace data for the given name.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_trace(trace_name, deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_trace_names(self, scintillator):
        """Returns a list of names of the traces that are currently being stored.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.

        Returns
        -------
        list
            A list of names of the traces that are currently being stored.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_trace_names()
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def find_matching_traces(self, scintillator, count_time, trace_list=None):
        """Finds the traces that could be a match for the given count (if they exist).

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.
        count_time : float
            The time that the count occurred at (in seconds of day).
        trace_list : list
            Optional. A list of traces that could be valid matches. If not provided, all traces stored for the
            scintillator of interest will be considered valid.

        Returns
        -------
        list
            A list of trace names that could be matches.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].find_matching_traces(count_time, self.date_str, trace_list)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def _projected_memory(self):
        """Returns the projected size (in bytes) of the object after all currently listed files have been imported."""
        total_file_size = 0
        for scintillator in self._scintillators:
            for file in self.get_attribute(scintillator, 'lm_filelist', deepcopy=False):
                total_file_size += tl.file_size(file) * self.lm_growth_factors[scintillator]

            for file in self.get_attribute(scintillator, 'trace_filelist', deepcopy=False):
                total_file_size += tl.file_size(file) * self.trace_growth_factors[scintillator]

        return total_file_size

    def clear(self, clear_filelists=True):
        """Clears all data currently stored in the Detector.

        Parameters
        ----------
        clear_filelists : bool
            Optional. If True, lists of files stored in the Detector will be also be cleared. True by default.

        """

        for scintillator in self._scintillators:
            self._scintillators[scintillator].clear(clear_filelists)

        if clear_filelists:
            self.dates_stored = [self.date_str]

    def _get_serial_num_filelist(self, eRC):
        """Returns a list of data files for the scintillator with the given eRC serial number."""
        complete_filelist = glob.glob(f'{self._import_loc}/{self.file_form(eRC)}')
        if len(complete_filelist) == 0:  # Here in case the data files are grouped into daily folders
            complete_filelist = glob.glob(f'{self._import_loc}/{self.date_str}/{self.file_form(eRC)}')

        if len(complete_filelist) == 0:  # Here in case the data files are grouped into non-daily folders
            complete_filelist = glob.glob(f'{self._import_loc}/*/{self.file_form(eRC)}')

        return complete_filelist

    def _get_scint_filelists(self, scintillator):
        """Returns the list mode and trace filelists for the given scintillator."""
        complete_filelist = self._get_serial_num_filelist(self._scintillators[scintillator].eRC)
        try:
            with open(f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config/detector_config.json',
                      'r') as file:
                entries = json.load(file)
        except json.decoder.JSONDecodeError:
            raise SyntaxError('invalid syntax in detector config file.')
        except KeyError:
            raise KeyError('missing valid file extension information in detector config file.')

        lm_extensions = entries['lm_extensions']
        trace_extensions = entries['trace_extensions']
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
        return lm_filelist, trace_filelist


    @staticmethod
    def _read_data_file(reader, filelist, clean_energy, connection):
        """Reading the given data files and sending them to the given pipe. Meant to be run in a subprocess, which 
        means that the passed arguments and piped values are serialized/deserialized on each end."""
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

    def _import_lm(self, process_pool, scintillator, options):
        """Imports list mode data for the given scintillator."""
        lm_filelist = self._scintillators[scintillator].lm_filelist
        file_frames = []
        file_ranges = []
        file_indices = {}
        start_index = 0
        log_strings = []

        if self.log is not None:
            log_strings.append('List Mode Files:\nFile|Import Success|\n')

        # Importing the data
        end1, end2 = multiprocessing.Pipe()
        file_index = 0
        process_pool.apply_async(self._read_data_file, args=(self._scintillators[scintillator].reader, lm_filelist,
                                                             options['clean_energy'], end2))
        while True:
            data = end1.recv()  # Reading data from the other process
            end1.send(1)  # Notifying the other process that the data has been received
            if isinstance(data, pd.DataFrame):
                pass
            elif isinstance(data, list):
                # Logging any warnings from the previous file
                log_strings.append('\t' + '\t'.join(data))
                continue
            elif isinstance(data, Reader):
                # Storing the updated reader object from the other process, which also serves as sentinel value
                self._scintillators[scintillator].reader = data
                break
            else:
                # Logging any reader errors
                log_strings.append(f'{lm_filelist[file_index]}|False|\n\tError importing file: {data}\n')
                file_index += 1
                continue

            first_second = data['SecondsOfDay'].min()
            last_second = data['SecondsOfDay'].max()

            # Determines the time gaps between adjacent files
            file_ranges.append([first_second, last_second])
            file_frames.append(data)
            if self.log is not None:
                log_strings.append(f'{lm_filelist[file_index]}|True|\n')

            # Keeps track of file indices in the larger dataframe
            data_length = len(data.index)
            file_indices[lm_filelist[file_index]] = [start_index, start_index + data_length]
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

            self._scintillators[scintillator].lm_frame = lm_frame
            self._scintillators[scintillator].lm_file_ranges = file_ranges
            self._scintillators[scintillator].lm_file_indices = file_indices

            if self.log is not None:
                log_strings.append(f'\nTotal Counts: {len(self._scintillators[scintillator].lm_frame.index)}\n\n')

        else:
            if self.log is not None:
                log_strings.append('\n')

        return len(file_frames), ''.join(log_strings)

    def _import_traces(self, process_pool, scintillator, options):
        """Imports trace data for the given scintillator."""
        trace_filelist = self._scintillators[scintillator].trace_filelist
        traces = {}
        log_strings = []
        if self.log is not None:
            log_strings.append('Trace Files:\nFile|Import Success|\n')

        # Importing the data
        end1, end2 = multiprocessing.Pipe()
        file_index = 0
        process_pool.apply_async(self._read_data_file, args=(self._scintillators[scintillator].reader, trace_filelist,
                                                             options['clean_energy'], end2))
        while True:
            data = end1.recv()  # Reading data from the other process
            end1.send(1)  # Notifying the other process that the data has been received
            if isinstance(data, pd.DataFrame):
                pass
            elif isinstance(data, list):
                # Logging any warnings from the previous file
                log_strings.append('\t' + '\t'.join(data))
                continue
            elif isinstance(data, Reader):
                # Storing the updated reader object from the other process, which also serves as sentinel value
                self._scintillators[scintillator].reader = data
                break
            else:
                # Logging any reader errors
                log_strings.append(f'{trace_filelist[file_index]}|False|\n\tError importing file: {data}\n')
                file_index += 1
                continue

            traces[trace_filelist[file_index]] = data
            if self.log is not None:
                log_strings.append(f'{trace_filelist[file_index]}|True|\n')

            file_index += 1

        # Storing the traces
        if len(traces) > 0:
            self._scintillators[scintillator].traces = traces

        if self.log is not None:
            log_strings.append('\n')

        return len(traces), ''.join(log_strings)

    def _import_scintillator(self, process_pool, output_lock, scintillator, options):
        """Manages data importing for a single scintillator. Meant to be run on a separate thread."""
        eRC = self._scintillators[scintillator].eRC
        lm_filelist_len = len(self._scintillators[scintillator].lm_filelist)
        trace_filelist_len = len(self._scintillators[scintillator].trace_filelist)
        # Writing initial status to stdout
        if options['feedback']:
            with output_lock:
                print('')
                print(f'For eRC {eRC} ({scintillator}):')
                if options['import_lm']:
                    if lm_filelist_len > 0:
                        print(f'Importing {lm_filelist_len} list mode files...')
                    else:
                        print('Missing list mode data')

                if options['import_traces']:
                    if trace_filelist_len > 0:
                        print(f'Importing {trace_filelist_len} trace files...')
                    else:
                        print('No trace data')

        if options['import_lm'] and lm_filelist_len > 0:
            lm_results = self._import_lm(process_pool, scintillator, options)
        else:
            lm_results = None

        if options['import_traces'] and trace_filelist_len > 0:
            trace_results = self._import_traces(process_pool, scintillator, options)
        else:
            trace_results = None

        if (lm_results is not None or trace_results is not None) and (options['feedback'] or self.log is not None):
            with output_lock:
                # Writing import status to stdout
                if options['feedback']:
                    print('')
                    print(f'For eRC {eRC} ({scintillator}):')
                    if options['import_lm'] and lm_results is not None:
                        print(f'{lm_results[0]}/{lm_filelist_len} list mode files imported')
                    else:
                        print('No list mode data imported')

                    if options['import_traces'] and trace_results is not None:
                        print(f'{trace_results[0]}/{trace_filelist_len} trace files imported')
                    else:
                        print(f'No traces imported')

                # Writing import information to the log
                if self.log is not None:
                    print(f'For eRC {eRC} ({scintillator}):', file=self.log)
                    if options['import_lm'] and lm_results is not None:
                        self.log.write(lm_results[1])

                    if options['import_traces'] and trace_results is not None:
                        self.log.write(trace_results[1])

    @staticmethod
    def _worker_init():
        """Data importer worker process initialization function. Mutes worker stdout and stderr output."""
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def import_data(self, existing_filelists=False, import_traces=True, import_lm=True, import_scints=None,
                    clean_energy=False, feedback=False, mem_frac=1.):
        """Imports and stores data from the daily data files.

        Parameters
        ----------
        existing_filelists : bool
            Optional. If True, the function will use the file lists already stored in the Detector.
        import_traces : bool
            Optional. If True, the function will import any trace files it finds. True by default.
        import_lm : bool
            Optional. If True, the function will import any list mode data it finds. True by default.
        import_scints : list
            Optional. If provided, only data from the scintillators with the listed names will be imported.
        clean_energy : bool
            Optional. If True, the data reader will strip out maximum and low energy counts. False by default.
        feedback : bool
            Optional. If True, feedback about the progress of the importing will be printed.
        mem_frac : float
            Optional: The maximum fraction of currently available system memory that the Detector is allowed to use for
            data. If the dataset is projected to be larger than this limit, a MemoryError will be raised.
            1.0 (100% of available system memory) by default.

        """

        if not self._has_identity:
            raise TypeError('cannot import data for a Detector with no identity.')

        if len(self.dates_stored) > 1:
            raise RuntimeError("cannot import multiple days' data.")

        if import_scints is None:
            scintillators = self.scint_list
        else:
            scintillators = []
            for scintillator in import_scints:
                if scintillator in self.scint_list:
                    scintillators.append(scintillator)

        if not existing_filelists:
            # Locates the files to be imported
            for scintillator in scintillators:
                lm_filelist, trace_filelist = self._get_scint_filelists(scintillator)
                if import_lm:
                    self._scintillators[scintillator].lm_filelist = lm_filelist

                if import_traces:
                    self._scintillators[scintillator].trace_filelist = trace_filelist

        # Checking to make sure that currently listed data won't go over the memory limit
        if self._projected_memory() > psutil.virtual_memory()[1] * mem_frac:
            raise MemoryError('dataset larger than specified limit.')

        if self.log is not None:
            print('', file=self.log)

        options = {'import_traces': import_traces, 'import_lm': import_lm, 'clean_energy': clean_energy,
                   'feedback': feedback}

        # Creating the process pool for data reading tasks
        with multiprocessing.Pool(processes=len(scintillators), initializer=self._worker_init) as process_pool:
            # Creating threads for each scintillator's import manager
            output_lock = threading.Lock()  # Mutex lock for log and stdout output
            threads = [threading.Thread(target=self._import_scintillator,
                                        args=(process_pool, output_lock, scintillator, options),
                                        daemon=True)
                       for scintillator in scintillators]

            for thread in threads:
                thread.start()

            # Waiting for data importing to finish for each scintillator
            for thread in threads:
                thread.join()

        gc.collect()

    def get_clone(self):
        """Returns a new Detector with the same identity as the current one (but no data).

        Returns
        -------
        tgfsearch.detectors.detector.Detector
            A new Detector with the same identity as the current one. Identity includes: unit name, date, print
            feedback setting, deployment info, default scintillator, import directory, export directory,
            scintillator configuration, and processed data flag.

        """

        clone = type(self)(self.unit, self.date_str)
        clone._import_loc = self._import_loc
        clone._results_loc = self._results_loc
        return clone

    def splice(self, operand_detector):
        """Returns a new Detector with the combined data of the current Detector and the one provided.

        Parameters
        ----------
        operand_detector : tgfsearch.detectors.detector.Detector
            Another Detector. Must have the same unit name and scintillator configuration as the current one.

        Returns
        -------
        tgfsearch.detectors.detector.Detector
            A new Detector containing the data of the current Detector and the one provided.

            Three things to note:
                - This Detector will store the date of the *earlier* operand Detector

                - All time-related data will be adjusted to reflect the difference in date between the operand Detectors

                - Using the method import() with this Detector won't work. Trying this will result in a RuntimeError

        """

        if operand_detector.unit == self.unit and operand_detector._scintillators.keys() == self._scintillators.keys():
            if int(self.date_str) < int(operand_detector.date_str):
                new_detector = self.get_clone()
                earlier = self
                later = operand_detector
            elif int(self.date_str) > int(operand_detector.date_str):
                new_detector = operand_detector.get_clone()
                earlier = operand_detector
                later = self
            else:
                raise ValueError('cannot splice the same day into itself.')

            new_detector.dates_stored.append(later.date_str)

            # Measuring the number of days between the earlier and later date
            day_difference = 0
            rolled_date = earlier.date_str
            while rolled_date != later.date_str:
                rolled_date = tl.roll_date_forward(rolled_date)
                day_difference += 1

            for scintillator in self._scintillators:
                # Combining list mode file lists
                new_detector.set_attribute(scintillator, 'lm_filelist',
                                           earlier.get_attribute(scintillator, 'lm_filelist') +
                                           later.get_attribute(scintillator, 'lm_filelist'),
                                           deepcopy=False)

                # Combining list mode file ranges
                # Updating ranges from later Detector to reflect the date difference
                new_ranges = later.get_attribute(scintillator, 'lm_file_ranges')
                for new_range in new_ranges:
                    new_range[0] += day_difference * params.SEC_PER_DAY
                    new_range[1] += day_difference * params.SEC_PER_DAY

                new_detector.set_attribute(scintillator, 'lm_file_ranges',
                                           earlier.get_attribute(scintillator, 'lm_file_ranges') +
                                           new_ranges, deepcopy=False)

                # Combining list mode file indices
                # Updating indices from later Detector to reflect their new positions in the data frame
                new_start = len(earlier.get_attribute(scintillator, 'lm_frame', deepcopy=False))
                new_indices = later.get_attribute(scintillator, 'lm_file_indices')
                for file in new_indices:
                    new_indices[file][0] += new_start
                    new_indices[file][1] += new_start

                new_indices.update(earlier.get_attribute(scintillator, 'lm_file_indices'))
                new_detector.set_attribute(scintillator, 'lm_file_indices', new_indices, deepcopy=False)

                # Combining list mode data frames
                # Updating later frame to reflect the difference in days
                later_frame = later.get_attribute(scintillator, 'lm_frame')
                later_frame['SecondsOfDay'] += day_difference * params.SEC_PER_DAY

                new_lm_frame = pd.concat([
                    earlier.get_attribute(scintillator, 'lm_frame', deepcopy=False),
                    later_frame], axis=0)

                new_detector.set_attribute(scintillator, 'lm_frame', new_lm_frame, deepcopy=False)

                # Combining trace file lists
                new_detector.set_attribute(scintillator, 'trace_filelist',
                                           earlier.get_attribute(scintillator, 'trace_filelist') +
                                           later.get_attribute(scintillator, 'trace_filelist'),
                                           deepcopy=False)

                # Combining trace tables
                new_traces = earlier.get_attribute(scintillator, 'traces')
                new_traces.update(later.get_attribute(scintillator, 'traces'))
                new_detector.set_attribute(scintillator, 'traces', new_traces, deepcopy=False)

            return new_detector
        else:
            raise TypeError(f"cannot splice '{self.unit}' ({self.scint_list}) with "
                            f"'{operand_detector.unit}' ({operand_detector.scint_list}).")
