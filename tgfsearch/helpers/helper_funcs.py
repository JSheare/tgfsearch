"""A module containing functions used by various parts of the package."""
import datetime as dt
import glob
import json
import os
import struct
from typing import Any, List


def is_valid_dir_path(path: str) -> bool:
    """Returns true if the given path is a valid directory path."""
    if os.path.exists(path):
        if os.path.isdir(path):
            return True

        return False

    try:
        os.mkdir(path)
        os.rmdir(path)
    except Exception:
        return False

    return True


def make_path(path: str) -> None:
    """Checks to see if a directory path corresponding to the given string exists and, if not, creates it.

    Parameters
    ----------
    path : str
        The path to be created.

    """

    if not os.path.exists(path):
        os.makedirs(path)


def file_size(file: str, uncompressed: bool = True) -> int:
    """Returns the size of the given file in bytes.

    Parameters
    ----------
    file : str
        The name of the file.
    uncompressed : bool
        Optional. If True, the function will return the uncompressed file size (if the file is compressed). True
        by default. Note: this will not be accurate for files that are over 4GB uncompressed due to the way that
        uncompressed size is stored according to the gzip standard.

    Returns
    -------
    int
        The size of the file in bytes.

    """

    if uncompressed and len(file) > 3 and file[-3:] == '.gz':
        with open(file, 'rb') as f:
            f.seek(-4, 2)
            size = struct.unpack('I', f.read(4))[0]

    else:
        size = os.path.getsize(file)

    return size


def read_json_file(file: str) -> Any:
    """A function that reads the given JSON file and returns it as the appropriate data structure.

    Parameters
    ----------
    file : str
        The file to be read.

    Returns
    -------
    any
        The json file's contents.

    """

    try:
        with open(file, 'r') as f:
            result = json.load(f)

        return result
    except json.decoder.JSONDecodeError:
        raise SyntaxError('invalid JSON syntax.')


def write_json_file(data: Any, file: str, pretty: bool = False) -> None:
        """A function that writes the given dictionary as JSON to the given file.

        Parameters
        ----------
        data : any
            The JSON-serializable data to be written.
        file : str
            The name of the file to write the dictionary to as JSON.
        pretty : bool
            Optional. If True, pretty prints the contents of the file.

        """

        with open(file, 'w') as f:
            if pretty:
                json.dump(data, f, indent=4)
            else:
                json.dump(data, f)


def yymmdd_to_date(date_str: str) -> dt.date:
    """Returns the given date string in YYMMDD format as a datetime.date object."""
    return dt.datetime.strptime(date_str, '%y%m%d').astimezone(dt.UTC).date()


def date_to_yymmdd(date: dt.date) -> str:
    """Returns the given datetime.date object as a string in the YYMMDD format."""
    return date.strftime('%y%m%d')


def get_date_list(date1: dt.date, date2: dt.date)-> List[dt.date]:
    """Returns a list of datetime.date objects on the given date range.

    Parameters
    ----------
    date1 : str
        The beginning of the date range.
    date2 : str
        The end (inclusive) of the date range.

    Returns
    -------
    list of datetime.date
        A list of datetime.date objects on the given date range.

    """

    dates = []
    while date1 != date2:
        dates.append(date1)
        date1 += dt.timedelta(days=1)

    dates.append(date2)
    return dates


def get_first_sec(date_str: str) -> float:
    """Returns the first epoch second of the given YYMMDD-format date."""
    date = dt.datetime.strptime(date_str, '%y%m%d').astimezone(dt.UTC)
    return (date - dt.timedelta(hours=date.hour, minutes=date.minute, seconds=date.second,
                                microseconds=date.microsecond)).timestamp()

def file_timestamp(file: str) -> str:
    """Returns the timestamp of the given data file as a string of the form hhmmss."""
    labels = file.split('.')[0].split('_')
    if len(labels) > 4:
        return labels[-2]

    return labels[-1]


def get_serial_num_filelist(import_loc: str, erc: str, date: dt.date) -> List[str]:
    """Returns a list of all data files with the given eRC serial number and date in the given import loc."""
    file_name_form = f'eRC{erc}*_*_{date_to_yymmdd(date)}_*'
    complete_filelist = glob.glob(f'{import_loc}/{file_name_form}')
    if len(complete_filelist) == 0:  # Here in case the data files are grouped into daily folders
        complete_filelist = glob.glob(f'{import_loc}/{date_to_yymmdd(date)}/{file_name_form}')

    if len(complete_filelist) == 0:  # Here in case the data files are grouped into non-daily folders
        complete_filelist = glob.glob(f'{import_loc}/*/{file_name_form}')

    return complete_filelist
