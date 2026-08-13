"""A module containing functions used by various parts of the package."""
import datetime
import json
import os
import struct
from typing import Any, List


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
    Any
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
        data : Any
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


def yymmdd_to_date(date_str: str) -> datetime.date:
    """Returns the given date string in YYMMDD format as a datetime.date object."""
    return datetime.datetime.strptime(date_str, '%y%m%d').astimezone(datetime.UTC).date()


def date_to_yymmdd(date: datetime.date) -> str:
    """Returns the given datetime.date object as a string in the YYMMDD format."""
    return date.strftime('%y%m%d')


def get_date_list(date_str_1: str, date_str_2: str)-> List[datetime.date]:
    """Returns a list of datetime.date objects on the given date_range.

    Parameters
    ----------
    date_str_1 : str
        The beginning of the date range as a date string in YYMMDD format.
    date_str_2 : str
        The end (inclusive) of the date range as a date string in YYMMDD format.

    Returns
    -------
    List[datetime.date]
        A list of datetime.date objects on the given date range.

    """

    start = yymmdd_to_date(date_str_1)
    end = yymmdd_to_date(date_str_2)
    dates = []
    while start != end:
        dates.append(start)
        start += datetime.timedelta(days=1)

    dates.append(end)
    return dates


def get_first_sec(date_str: str) -> float:
    """Returns the first epoch second of the given YYMMDD-format date."""
    date = datetime.datetime.strptime(date_str, '%y%m%d').astimezone(datetime.UTC)
    return (date - datetime.timedelta(hours=date.hour, minutes=date.minute, seconds=date.second,
                                      microseconds=date.microsecond)).timestamp()
