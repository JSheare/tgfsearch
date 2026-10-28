"""A module containing a base class for keeping track of instrument data and associated information."""
from __future__ import annotations

import copy
import datetime as dt
import gc
import glob
import logging
import multiprocessing
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
import pickle
import psutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
from typing import Any, Dict, Generator, List

import tgfsearch.config.parameters as params
import tgfsearch.helpers.helper_funcs as helper_funcs
from tgfsearch.instruments.scintillator import Scintillator
from tgfsearch.tools.reader import Reader


def _worker_init() -> None:
    """Data importer worker process initialization function. Mutes worker stdout and stderr output."""
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


class Instrument:
    """A class used to store all relevant information about an instrument and its data for a day.

    The Instrument class is used to store the name of the instrument, the date, and the actual data for the requested
    day in a single, centralized location.

    Parameters
    ----------
    name : str
        The name of the instrument.
    date_str : str
        The timestamp for the requested day in yymmdd format.

    Attributes
    ----------
    date : datetime.date
        The date that the Instrument holds data for. If the Instrument holds data for multiple dates, it will be the
        earliest one.
    first_sec : float
        The first second of the day in EPOCH time.
    dates_stored : list of datetime.date
        A list of dates currently being stored in the Instrument.
    _has_identity : bool
        A flag for whether the Instrument has an identity (established name, scintillator configuration, etc.).
    _import_loc : str
        The directory where data files for the day are located.
    _scintillators : dict of [str, tgfsearch.instruments.scintillator.Scintillator]
        A dictionary containing Scintillators. These keep track of data for each of the instrument's scintillators.
        Note the name mangling underscore.
    _scint_list : list of str
        A list of the instrument's scintillator names in descending order of priority.
    deployment : dict of [str, any]
        Deployment information for the instrument on the requested day (if available).

    """

    def __init__(self, name: str, date_str: str, **kwargs) -> None:
        # Basic information
        self.date = helper_funcs.yymmdd_to_date(date_str)
        self.first_sec = helper_funcs.get_first_sec(date_str)
        self.dates_stored = [self.date]

        # Identity-related information
        self._has_identity = False
        self.name = name.upper()
        self._import_loc = ''
        self._scintillators = {}
        self._scint_list = []
        self.deployment = self._get_deployment()

        # Allows us to disable identity reading if we need to, but without advertising it in the documentation
        if 'read_identity' in kwargs and not kwargs['read_identity']:
            pass
        else:
            self._read_identity()

    def __del__(self) -> None:
        self.clear()

    def __str__(self) -> str:
        """String casting overload. Returns a string of the form 'Instrument(name, date_yymmdd)'."""
        return f'Instrument({self.name}, {helper_funcs.date_to_yymmdd(self.date)})'

    def __repr__(self) -> str:
        """Debugging string magic method. Returns a string of the form 'Instrument(name, date_yymmdd)' along
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

    def __iter__(self) -> Generator[str, None, None]:
        """Iterator magic method. Returns a generator that yields the Instrument's scintillator names in descending
        order of priority."""
        for scintillator in self._scint_list:
            yield scintillator

    def __bool__(self) -> bool:
        """Bool casting overload. Returns True if data for any scintillator is present."""
        if self._has_identity:
            for scintillator in self._scintillators:
                if self.data_present_in(scintillator):
                    return True

        return False

    def __contains__(self, scintillator) -> bool:
        """Contains overload. Returns True if the provided string corresponds to a scintillator in Instrument,
        False otherwise."""
        return scintillator in self._scintillators

    def __add__(self, operand_instrument) -> Instrument:
        """Addition operator overload. Returns a new Instrument containing data from the current Instrument
        and the provided one. See splice() method documentation for more information."""
        return self.splice(operand_instrument)

    def _get_deployment(self) -> Dict[str, Any]:
        """Returns a dictionary full of deployment information for the instrument on its specified date."""
        for file in glob.glob(
                f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/deployments/'
                f'{self.name}_deployment_*.json'):
            file_dates = file.split('deployment')[-1][1:].split('.')[0].split('_')
            start_date = helper_funcs.yymmdd_to_date(file_dates[0])
            end_date = helper_funcs.yymmdd_to_date(file_dates[1])
            if start_date <= self.date <= end_date:
                return helper_funcs.read_json_file(file)

        return {'location': 'no location listed', 'instrument': self.name, 'start_date': '000000', 'end_date': '000000',
                'tz_identifier': '', 'weather_station': '', 'sounding_station': '', 'latitude': 0., 'longitude': 0.,
                'altitude': 0., 'notes': ''}

    def _read_identity(self) -> None:
        """Gets and fills in the identity of the Instrument from the config file based on the name and date."""
        try:
            entries = helper_funcs.read_json_file(
                f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config/instrument_config.json')
        except SyntaxError:
            raise SyntaxError('invalid syntax in instrument config file.')

        if self.name in entries['identities']:
            identity = entries['identities'][self.name]
            try:
                self._import_loc = (f'{entries["default_data_root"]}/{identity["subdir"]}/'
                                    f'{helper_funcs.date_to_yymmdd(self.date)}')

                # Getting the right scintillator configuration based on the date
                correct_date_str = ''
                for after_date_str in identity['scintillators']:
                    if after_date_str == '000000':
                        after_date = dt.datetime.fromtimestamp(0, dt.UTC).date()
                    else:
                        after_date = dt.datetime.strptime(after_date_str, '%y%m%d').astimezone(dt.UTC).date()

                    if self.date >= after_date:
                        correct_date_str = after_date_str
                    else:
                        break

                # Putting the scintillators in the correct order of priority
                for scintillator in identity['scintillators'][correct_date_str]:
                    self._scint_list.append(scintillator)

                self._scint_list.sort(key=lambda scint: entries['scintillator_priority'].index(scint))

                # Creating the scintillator objects
                for scintillator in self._scint_list:
                    scint_entry = identity['scintillators'][correct_date_str][scintillator]
                    self._scintillators[scintillator] = Scintillator(scintillator, scint_entry['erc'], self.date,
                                                                     scint_entry['format_name'])

                self._has_identity = True
            except KeyError:
                raise SyntaxError(f"missing or incomplete information in instrument config file for '{self.name}'.")

        else:
            raise ValueError(f"'no entry for '{self.name}' in instrument config file.")

    def pickle(self, file_name: str, path: str) -> str:
        """Pickles the instrument instance to a file and returns the path.

        Parameters
        ----------
        file_name : str
            The name of the pickle file.
        path : str
            The directory where the pickle file will be saved.

        Returns
        -------
        str
            The path to the pickle file (including its name).

        """

        helper_funcs.make_path(path)

        export_path = f'{path}/{file_name}.pkl'
        with open(export_path, 'wb') as file:
            pickle.dump(self, file)

        return export_path

    @staticmethod
    def unpickle(pickle_path: str) -> Instrument:
        """Unpickles an Instrument from a file and returns it.

        Parameters
        ----------
        pickle_path : str
            The path (including file name) to the pickle file that the Instrument is stored in.

        Returns
        -------
        tgfsearch.instruments.instrument.Instrument
            A Instrument.

        """

        with open(pickle_path, 'rb') as file:
            instrument = pickle.load(file)

        return instrument

    def has_identity(self) -> bool:
        """Returns True if the Instrument has an established identity (established name, scintillator configuration,
        etc.), False otherwise."""
        return self._has_identity

    def get_import_loc(self) -> str:
        """Returns the directory where data will be imported from.

        Returns
        -------
        str
            The import directory as a string.

        """

        return self._import_loc

    def set_import_loc(self, loc: str) -> None:
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

    def is_named(self, name: str) -> bool:
        """Returns True if the Instrument has the same name as the passed string.

        Parameters
        ----------
        name : str
            The name being queried.

        Returns
        -------
        bool
            True if the name matches the name of the Instrument, False otherwise.

        """

        return name.upper() in self.name

    def is_valid_scintillator(self, scintillator: str) -> bool:
        """Returns True if the Instrument contains the specified scintillator, False otherwise

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.

        Returns
        -------
        bool
            True if the scintillator is in Instrument, False otherwise.

        """

        return scintillator in self._scintillators

    def data_present_in(self, scintillator: str, data_type: str = 'lm') -> bool:
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

        Raises
        ------
        ValueError
            If either the passed scintillator or the passed data type aren't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].data_present(data_type)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_scintillator_identity(self, scintillator: str) -> Dict[str, str | dt.date]:
        """Returns a dictionary containing identifying information about the given scintillator.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.

        Returns
        -------
        dict of [str, str or datetime.date]
            The scintillator's identifying information, including: its name, its eRC serial number, its data date, its
            list mode data format name, and more.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_identity()
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_reader(self, scintillator: str) -> Reader:
        """Returns the Reader object for the given scintillator.

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.

        Returns
        -------
        tgfsearch.tools.reader.Reader
            The scintillator's Reader object.

        Raises
        ------
        ValueError
            If either the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_reader()
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def set_reader(self, scintillator: str, reader: Reader) -> None:
        """Sets the given scintillator's reader to the given Reader object.

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.
        reader : tgfsearch.tools.reader.Reader
            The Reader object to be set.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            self._scintillators[scintillator].set_reader(reader)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_lm_filelist(self, scintillator: str, deepcopy: bool = True) -> List[str]:
        """Returns the list mode file list for the given scintillator. Note that, if not set explicitly,  the list mode
        data file list is generated lazily (i.e. not until data is imported).

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.
        deepcopy : bool
            Optional. If True, a copy of the file list will be returned instead of the original. True by default.

        Returns
        -------
        list of str
            The list of list mode file names for the given scintillator.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_lm_filelist(deepcopy=deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def set_lm_filelist(self, scintillator: str, lm_filelist: List[str], deepcopy: bool = True) -> None:
        """Sets the given scintillator's list mode file list to the given list.

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.
        lm_filelist : list of str
            The new list mode file list.
        deepcopy : bool
            Optional. If True, the scintillator's list mode file list will be set to a copy of the given list. True
            by default.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            self._scintillators[scintillator].set_lm_filelist(lm_filelist, deepcopy=deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_lm_data(self, scintillator: str, column: str, file_name: str | None = None) -> npt.NDArray[np.float64]:
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

        Raises
        ------
        ValueError
            If either the passed scintillator or the passed column aren't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_lm_data(column, file_name)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def set_lm_data(self, scintillator: str, column: str, new_data: npt.NDArray[np.float64],
                    file_name: str | None = None) -> None:
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

        Raises
        ------
        ValueError
            If either the passed scintillator or the passed column aren't valid, or if the length of the new data
            doesn't match (the length of the file or whole dataset depending on whether a file name was given).

        """

        if scintillator in self._scintillators:
            self._scintillators[scintillator].set_lm_data(column, new_data, file_name)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def find_lm_file(self, scintillator: str, count_time: float) -> str:
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

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].find_lm_file(count_time)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_lm_frame(self, scintillator: str, file_name: str | None = None, deepcopy: bool = True) -> pd.DataFrame:
        """Returns the list mode data frame for the given scintillator.

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.
        file_name : str
            Optional. The name of the file that the frame is being requested for. Note that this must be the *full* name
            of the file, including the path from the root directory. Also note that, because of how DataFrames work,
            this will make the deepcopy parameter implicitly True.
        deepcopy : bool
            Optional. If True, a copy of the data frame will be returned instead of the original. True by default.

        Returns
        -------
        pandas.core.frame.DataFrame
            The list mode data frame for the given scintillator.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid or if the passed file name isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_lm_frame(file_name=file_name, deepcopy=deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def set_lm_frame(self, scintillator: str, lm_frame: pd.DataFrame, file_name: str | None = None,
                     deepcopy: bool = True) -> None:
        """Sets the given scintillator's trace file list to the given list.

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.
        lm_frame : pandas.core.frame.DataFrame
            The list mode data frame.
        file_name : str
            Optional. The name of the file that the frame is being set for. Note that this must be the *full* name of
            the file, including the path from the root directory. Also note that, because of how DataFrames work, this
            will make the deepcopy parameter implicitly True.
        deepcopy : bool
            Optional. If True, the scintillator's list mode data frame will be set to a copy of the given data frame.
            True by default.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid, the passed lm_frame isn't valid, or the passed file name isn't
            valid.

        """

        if scintillator in self._scintillators:
            self._scintillators[scintillator].set_lm_frame(lm_frame, file_name=file_name, deepcopy=deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_trace_filelist(self, scintillator: str, deepcopy: bool = True) -> List[str]:
        """Returns the trace file list for the given scintillator. Note that, if not set explicitly, the trace data file
        list is generated lazily (i.e. not until data is imported).

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.
        deepcopy : bool
            Optional. If True, a copy of the file list will be returned instead of the original. True by default.

        Returns
        -------
        list of str
            The list of trace file names for the given scintillator.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_trace_filelist(deepcopy=deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def set_trace_filelist(self, scintillator: str, trace_filelist: List[str], deepcopy: bool = True) -> None:
        """Sets the given scintillator's trace file list to the given list.

        Parameters
        ----------
        scintillator : str
            The scintillator of interest.
        trace_filelist : list of str
            The new trace file list.
        deepcopy : bool
            Optional. If True, the scintillator's trace file list will be set to a copy of the given list. True
            by default.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            self._scintillators[scintillator].set_trace_filelist(trace_filelist, deepcopy=deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_trace_names(self, scintillator: str) -> List[str]:
        """Returns a list of names of the traces that are currently being stored.

        Parameters
        ----------
        scintillator : str
            The name of the scintillator of interest.

        Returns
        -------
        list of str
            A list of names of the traces that are currently being stored.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_trace_names()
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def get_trace(self, scintillator: str, trace_name: str, deepcopy: bool = True) -> pd.DataFrame:
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

        Raises
        ------
        ValueError
            If either the passed scintillator or the passed file name aren't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].get_trace(trace_name, deepcopy)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def find_matching_traces(self, scintillator: str, count_time: float,
                             trace_list: List[str] | None = None) -> List[str]:
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
        list of str
            A list of trace names that could be matches.

        Raises
        ------
        ValueError
            If the passed scintillator isn't valid.

        """

        if scintillator in self._scintillators:
            return self._scintillators[scintillator].find_matching_traces(count_time, self.date.strftime('%y%m%d'),
                                                                          trace_list)
        else:
            raise ValueError(f"'{scintillator}' is not a valid scintillator.")

    def clear(self, clear_filelists: bool = True) -> None:
        """Clears all data currently stored in the Instrument.

        Parameters
        ----------
        clear_filelists : bool
            Optional. If True, lists of files stored in the Instrument will be also be cleared. True by default.

        """

        for scintillator in self._scintillators:
            self._scintillators[scintillator].clear(clear_filelists)

        if clear_filelists:
            self.dates_stored = [self.date]

    def __import_scintillator(self, process_pool: Pool, output_lock: threading.Lock, scintillator: str,
                              options: Dict[str, bool]) -> None:
        """Manages data importing for a single scintillator. Meant to be run on a separate thread."""
        erc = self._scintillators[scintillator].erc
        lm_filelist_len = len(self._scintillators[scintillator].get_lm_filelist(deepcopy=False))
        trace_filelist_len = len(self._scintillators[scintillator].get_trace_filelist(deepcopy=False))
        # Writing initial status to stdout
        if options['feedback']:
            with output_lock:
                print('')
                print(f'For eRC {erc} ({scintillator}):')
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
            lm_results = self._scintillators[scintillator].import_lm(process_pool, options)
        else:
            lm_results = None

        if options['import_traces'] and trace_filelist_len > 0:
            trace_results = self._scintillators[scintillator].import_traces(process_pool, options)
        else:
            trace_results = None

        if lm_results is not None or trace_results is not None:
            with output_lock:
                # Writing import status to stdout
                if options['feedback']:
                    print('')
                    print(f'For eRC {erc} ({scintillator}):')
                    if options['import_lm'] and lm_results is not None:
                        print(f'{lm_results[0]}/{lm_filelist_len} list mode files imported')
                    else:
                        print('No list mode data imported')

                    if options['import_traces'] and trace_results is not None:
                        print(f'{trace_results[0]}/{trace_filelist_len} trace files imported')
                    else:
                        print(f'No traces imported')

                # Writing import information to the log
                logger = logging.getLogger(params.PACKAGE_NAME)
                logger.info(f'For eRC {erc} ({scintillator}):')
                if options['import_lm'] and lm_results is not None:
                    logger.info(lm_results[1])

                if options['import_traces'] and trace_results is not None:
                    logger.info(trace_results[1])

    def import_data(self, existing_filelists: bool = False, import_traces: bool = True, import_lm: bool = True,
                    import_scints: List[str] | None = None, clean_energy: bool = False, mem_frac: float = 1.,
                    **kwargs) -> None:
        """Imports and stores data from the daily data files.

        Parameters
        ----------
        existing_filelists : bool
            Optional. If True, the function will use the file lists already stored in the Instrument.
        import_traces : bool
            Optional. If True, the function will import any trace files it finds. True by default.
        import_lm : bool
            Optional. If True, the function will import any list mode data it finds. True by default.
        import_scints : list of str
            Optional. If provided, only data from the scintillators with the listed names will be imported.
        clean_energy : bool
            Optional. If True, the data reader will strip out maximum and low energy counts. False by default.
        mem_frac : float
            Optional: The maximum fraction of currently available system memory that the Instrument is allowed to use
            for data. If the dataset is projected to be larger than this limit, a MemoryError will be raised.
            1.0 (100% of available system memory) by default.

        Raises
        ------
        TypeError
            If the Instrument instance doesn't have an identity.
        RuntimeError
            If the Instrument describes/contains multiple days' info.
        MemoryError
            If the projected memory usage after importing exceeds the given threshold.

        """

        if not self._has_identity:
            raise TypeError('cannot import data for an Instrument with no identity.')

        if len(self.dates_stored) > 1:
            raise RuntimeError("cannot import multiple days' data.")

        if import_scints is None:
            scintillators = self._scint_list
        else:
            scintillators = []
            for scintillator in import_scints:
                if scintillator in self._scintillators:
                    scintillators.append(scintillator)

        if not existing_filelists:
            # Locates the files to be imported
            for scintillator in scintillators:
                self._scintillators[scintillator].assemble_filelists(self._import_loc)

        # Checking to make sure that currently listed data won't go over the memory limit
        projected_memory = 0
        for scintillator in scintillators:
            projected_memory += self._scintillators[scintillator].get_projected_memory(include_lm=import_lm,
                                                                                       include_traces=import_traces)
        if projected_memory > psutil.virtual_memory()[1] * mem_frac:
            raise MemoryError('dataset larger than specified limit.')

        logging.getLogger(params.PACKAGE_NAME).info('Importing data:')

        options = {'import_traces': import_traces, 'import_lm': import_lm, 'clean_energy': clean_energy,
                   'feedback': kwargs['feedback'] if 'feedback' in kwargs else False}

        # Creating the process pool for data reading tasks
        with multiprocessing.Pool(processes=len(scintillators), initializer=_worker_init) as process_pool:
            # Creating threads for each scintillator's import manager
            output_lock = threading.Lock()  # Mutex lock for log and stdout output
            with ThreadPoolExecutor(max_workers=len(scintillators)) as executor:
                futures = [executor.submit(self.__import_scintillator,
                                           process_pool, output_lock, scintillator, options)
                           for scintillator in scintillators]

                # Waiting for data importing to finish for each scintillator
                for future in futures:
                    future.result()

        gc.collect()

    def get_clone(self) -> Instrument:
        """Returns a new Instrument with the same identity as the current one (but no data).

        Returns
        -------
        tgfsearch.instruments.instrument.Instrument
            A new Instrument with the same identity as the current one. Identity includes: instrument name, date,
            deployment info, import directory, and scintillator configuration.

        """

        clone = type(self)(self.name, helper_funcs.date_to_yymmdd(self.date))
        clone._import_loc = self._import_loc
        return clone

    def splice(self, operand_instrument: Instrument) -> Instrument:
        """Returns a new Instrument with the combined data of the current Instrument and the one provided.

        Parameters
        ----------
        operand_instrument : tgfsearch.instruments.instrument.Instrument
            Another Instrument. Must have the same instrument name and scintillator configuration as the current one.

        Returns
        -------
        tgfsearch.instruments.instrument.Instrument
            A new Instrument containing the data of the current Instrument and the one provided.

            Three things to note:
                - This Instrument will store the date of the *earlier* operand Instrument

                - All time-related data will be adjusted to reflect the difference in date between the operand
                  Instruments

                - Using the method import() with this Instrument won't work. Trying this will result in a RuntimeError

        Raises
        ------
        TypeError
            If the Instruments' names and configurations don't match.
        ValueError
            If the Instruments both describe the same day.

        """

        if (operand_instrument.name == self.name and
                operand_instrument._scintillators.keys() == self._scintillators.keys()):
            if self.date < operand_instrument.date:
                earlier = self
                later = operand_instrument
            elif self.date > operand_instrument.date:
                earlier = operand_instrument
                later = self
            else:
                raise ValueError('cannot splice the same day into itself.')

            new_instrument = type(self)(earlier.name, helper_funcs.date_to_yymmdd(earlier.date), read_identity=False)
            new_instrument.dates_stored.append(copy.deepcopy(later.date))

            for scintillator in earlier._scintillators:
                new_instrument._scint_list.append(scintillator)
                new_instrument._scintillators[scintillator] = (
                    earlier._scintillators[scintillator].splice(later._scintillators[scintillator]))

            new_instrument._has_identity = earlier._has_identity
            return new_instrument
        else:
            raise TypeError(f"cannot splice '{self.name}' ({self._scint_list}) with "
                            f"'{operand_instrument.name}' ({operand_instrument._scint_list}).")
