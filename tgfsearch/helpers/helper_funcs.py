"""A module containing functions used by various parts of the package."""
import os
import struct

import tgfsearch.config.parameters as params


def make_path(path):
    """Checks to see if a directory path corresponding to the given string exists and, if not, creates it.

    Parameters
    ----------
    path : str
        The path to be created.

    """

    if not os.path.exists(path):
        os.makedirs(path)


def file_size(file, uncompressed=True):
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


def days_per_month(month, year):
    """Returns the number of days in the requested month based on the year.

    Parameters
    ----------
    month : int
        The month of the year (1-12).
    year : int
        The desired year to check.

    Returns
    -------
    int
        The number of days in the specified month for the specified year.

    """

    match month:
        case 1:  # January
            return 31
        case 2:  # February
            return 29 if year % 4 == 0 or (year % 100 != 0 and year % 400 == 0) else 28
        case 3:  # March
            return 31
        case 4:  # April
            return 30
        case 5:  # May
            return 31
        case 6:  # June
            return 30
        case 7:  # July
            return 31
        case 8:  # August
            return 31
        case 9:  # September
            return 30
        case 10:  # October
            return 31
        case 11:  # November
            return 30
        case 12:  # December
            return 31
        case _:
            return 0


def roll_date_forward(date_str):
    """Returns the calendar date after the one given as an argument.

    Parameters
    ----------
    date_str : str
        The date to be rolled forward from (in yymmdd format).

    Returns
    -------
    str
        The calendar date after the one supplied (in yymmdd format).

    """

    date_int = int(date_str)
    date_int += 1
    date_str = str(date_int)
    # Month rollover
    if int(date_str[4:]) > days_per_month(int(date_str[2:4]), int(params.CENTURY + date_str[0:2])):
        date_int = date_int + 100 - (int(date_str[4:]) - 1)
        date_str = str(date_int)

    # Year rollover
    if int(date_str[2:4]) > 12:
        date_int = (date_int // 10000 + 1) * 10000 + 101
        date_str = str(date_int)

    return date_str


def roll_date_backward(date_str):
    """Returns the calendar date before the one given as an argument.

    Parameters
    ----------
    date_str : str
        The date to be rolled backward from (in yymmdd format).

    Returns
    -------
    str
        The calendar date before the one supplied (in yymmdd format).

    """

    date_int = int(date_str)
    date_int -= 1
    date_str = str(date_int)
    # Year rollback
    if int(date_str[2:]) == 100:  # This would be January 0th because int(0100) = 100
        date_int = (date_int // 10000 - 1) * 10000 + 1231  # December 31st of the previous year
        date_str = str(date_int)

    # Month rollback
    if int(date_str[4:]) == 0:
        date_int -= 100
        date_int += days_per_month(int(str(date_int)[2:4]), int(params.CENTURY + date_str[0:2]))
        date_str = str(date_int)

    return date_str


def make_date_list(first_date, second_date):
    """Makes a list of dates from first_date to second_date (inclusive).

    Parameters
    ----------
    first_date : str
        The first date in the range.
    second_date : str
        The second date in the range.

    Returns
    -------
    list
        A list of dates on the specified range.

    """

    requested_dates = [first_date]
    if first_date != second_date:
        date_str = first_date
        while True:
            date_str = roll_date_forward(date_str)
            requested_dates.append(date_str)
            if date_str == second_date:
                break

    return requested_dates