"""Tools for use by the TGF search program and its modules."""
import datetime as dt
import numpy as np
import os as os
import pandas as pd
import pickle as pickle
import struct as struct
from selenium import webdriver as webdriver
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

import tgfsearch.parameters as params


def o1_poly(x, a=0., b=0.):
    """Returns the y value at the given x for a first-order polynomial with terms a and b."""
    return x * a + b


def o2_poly(x, a=0., b=0., c=0.):
    """Returns the y value at the given x for a second-order polynomial with terms a, b, and c."""
    return a * x ** 2 + b * x + c


def print_logger(string, logfile):
    """Prints the specified string to both stdout and the specified file.

    Parameters
    ----------
    string : str
        The string to be printed/logged.
    logfile : _io.TextIO
        The file where the string should be written.

    """

    print(string)
    if logfile is not None:
        print(string, file=logfile)


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
        by default. Note: this will not be accurate for files that are over 4GB uncompressed due to the way
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


def file_timestamp(file):
    """Returns the timestamp of the given file as a string of the form hhmmss."""

    labels = file.split('.')[0].split('_')
    if len(labels) > 4:
        return labels[-2]

    return labels[-1]


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


def full_date_to_short(full_date_str):
    """Converts a date string of the form yyyy-mm-dd to the form yymmdd."""
    return full_date_str[2:].replace('-', '')


def short_to_full_date(date_str):
    """Converts a date string of the form yymmdd to the form yyyy-mm-dd."""
    return f'{params.CENTURY}{date_str[0:2]}-{date_str[2:4]}-{date_str[4:]}'


def get_first_sec(date_str):
    """Converts the given date string (in yymmdd format) to its first second in EPOCH time."""
    day = int(date_str[4:])
    month = int(date_str[2:4])
    year = int(params.CENTURY + date_str[0:2])
    return (dt.datetime(year, month, day, 0, 0) - dt.datetime(1970, 1, 1)).total_seconds()


def pickle_detector(detector, file_name, path=None):
    """Pickles Detectors.

    Parameters
    ----------
    detector : tgfsearch.detectors.detector.Detector
        The Detector to be pickled.
    file_name : str
        The name of the pickle file.
    path : str
        Optional. The directory where the pickle file will be saved. If not provided, the file will be saved
            to the Detector's daily results directory.

    Returns
    -------
    str
        The path to the pickle file (including its name).

    """

    if path is None:
        path = f'{detector.get_results_loc()}'

    make_path(path)

    log = detector.log
    detector.log = None  # serializing open file objects results in errors
    export_path = f'{path}/{file_name}.pickle'
    with open(export_path, 'wb') as file:
        pickle.dump(detector, file)

    detector.log = log
    return export_path


def unpickle_detector(pickle_path):
    """Unpickles Detectors.

    Parameters
    ----------
    pickle_path : str
        The path (including file name) to the pickle file that the Detector is stored in.

    Returns
    -------
    tgfsearch.detectors.detector.Detector
        A Detector.

    """

    with open(pickle_path, 'rb') as file:
        detector = pickle.load(file)

    return detector


def pickle_chunk(chunk, file_name):
    """Pickles daily chunks for the program's low memory mode."""
    return pickle_detector(chunk, file_name)


def unpickle_chunk(chunk_path):
    """Unpickles and loads daily chunks for the program's low memory mode."""
    with open(chunk_path, 'rb') as file:
        chunk = pickle.load(file)

    return chunk


def filter_data_files(complete_filelist):
    """Returns an ordered list of data files with duplicate/invalid files filtered out."""
    valid_filetypes = ['.txt', '.txt.gz', '.xtr', '.xtr.gz', '.csv', '.csv.gz']
    unique_files = set()
    file_names = []
    extensions = []
    for file in complete_filelist:
        if len(file) >= 4:
            dot_index = len(file) - 4 if file[-3:] == '.gz' else len(file) - 1
            while file[dot_index] != '.' and dot_index >= 0:
                dot_index -= 1

            full_extension = file[dot_index:]
            if full_extension in valid_filetypes:
                file_name = file[:dot_index]
            else:
                continue

            if file_name not in unique_files:
                unique_files.add(file_name)
                file_names.append(file_name)
                extensions.append(full_extension)

    return [file_names[s] + extensions[s] for s in np.argsort(file_names)]  # Puts the files back in order


def separate_data_files(filelist):
    """Returns a pair of ordered lists: one with list mode files, the other with trace files."""
    lm_filetypes = ['.txt', '.txt.gz', '.csv', '.csv.gz']
    lm_filelist = []
    trace_filelist = []
    for file in filelist:
        if len(file) >= 4:
            dot_index = len(file) - 4 if file[-3:] == '.gz' else len(file) - 1
            while file[dot_index] != '.' and dot_index >= 0:
                dot_index -= 1

            full_extension = file[dot_index:]
            if full_extension in lm_filetypes:
                lm_filelist.append(file)
            else:
                trace_filelist.append(file)

    return lm_filelist, trace_filelist


def convert_to_local(detector, event_time):
    """Converts the Detector date and event time to what they would actually be in local time.

    Parameters
    ----------
    detector : tgfsearch.detectors.detector.Detector
        The Detector where the date is stored.
    event_time : float
        The time of the day when the event occurred in seconds since the beginning of the day.

    Returns
    -------
    tuple[str, float]
        An updated date in local time and an updated event time in local time.

    """

    date_str = detector.date_str
    timezone_conversion = detector.deployment['utc_to_local']

    # Just in case the event happened in the ~300 seconds of the next day typically included in the dataset
    if event_time > params.SEC_PER_DAY:
        event_time -= params.SEC_PER_DAY
        date_str = roll_date_forward(date_str)

    # Corrects the UTC conversion if we're in daylight savings time
    if detector.deployment['dst_in_region']:
        timezone_conversion = dst_conversion(date_str, event_time, timezone_conversion)

    # If the event happened the next day local time
    if (event_time + (params.SEC_PER_HOUR * timezone_conversion)) > params.SEC_PER_DAY:
        date_str = roll_date_forward(date_str)
        event_time = (event_time + (params.SEC_PER_HOUR * timezone_conversion)) - params.SEC_PER_DAY
    # If the event happened the previous day local time
    elif (event_time + (params.SEC_PER_HOUR * timezone_conversion)) < 0:
        date_str = roll_date_backward(timezone_conversion)
        event_time = (event_time + (params.SEC_PER_HOUR * timezone_conversion)) + params.SEC_PER_DAY
    else:
        event_time = event_time + (params.SEC_PER_HOUR * timezone_conversion)

    return short_to_full_date(date_str), event_time


def scrape_weather(full_date_str, station):
    """Scrapes weather from weather underground and returns the results as a pandas data frame.

    Parameters
    ----------
    full_date_str : str
        The date that weather data is being requested for in yyyy-mm-dd format.
    station : str
        The four-letter name of the weather station that data is being requested for.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with weather information for the specified day.

    """

    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Runs chrome in headless mode (no browser tab)
        # The below options prevent an annoying logging entry from being printed to stdout
        chrome_options.add_experimental_option("excludeSwitches", ['enable-logging'])
        chrome_options.add_argument("--log-level=3")
        chrome_options.set_capability("browserVersion", "117")
        chrome_options.add_argument("start-maximized")

        driver = webdriver.Chrome(options=chrome_options)

        url = f'https://www.wunderground.com/history/daily/{station.upper()}/date/{full_date_str}'
        driver.get(url)
        tables = WebDriverWait(driver, 20).until(ec.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))

        table = pd.read_html(tables[1].get_attribute('outerHTML'))[0]

        return table.dropna()  # This is a dataframe containing the table we want
    except:
        return pd.DataFrame()


def get_weather_conditions(detector, full_date_str, event_time, weather_cache=None):
    """Scrapes weather underground and returns the weather around the time of an event.

    Parameters
    ----------
    detector : tgfsearch.detectors.detector.Detector
        The Detector that contains the name of the nearest weather station.
    full_date_str : str
        The date that the event occurred on (in local time) in yyyy-mm-dd format.
    event_time : float
        The time that the event occurred at during the day (in local time) in units of seconds since beginning of day.
    weather_cache : dict
       Optional. A cache containing weather tables that have already been retrieved. The keys are dates in
       yyyy-mm-dd format.

    Returns
    -------
    str
        The weather conditions around the time of the event as a string.

    """

    if weather_cache is None:
        weather_cache = {}

    if full_date_str in weather_cache and weather_cache[full_date_str] is not None:
        weather_table = weather_cache[full_date_str]
    else:
        weather_table = scrape_weather(full_date_str, detector.deployment['weather_station'])
        weather_cache[full_date_str] = weather_table

    # If changing weather scores here remember to update them in weather_from_score below
    if not weather_table.empty:
        # Finds the time in the table that's closest to the time of the event
        index = 0
        best_diff = float('inf')
        best_index = 0
        for clock_hour in weather_table['Time']:
            if not isinstance(clock_hour, float):
                time_sec = convert_clock_hour(clock_hour)
                time_diff = abs(event_time - time_sec)
                if time_diff < best_diff:
                    best_diff = time_diff
                    best_index = index
            else:
                break

            index += 1

        # Gets the weather conditions at the closest hour to the event and the surrounding hour_padding hours
        weather = []
        for i in range(best_index - params.WEATHER_PADDING, best_index + params.WEATHER_PADDING + 1):
            if 0 <= i < index:
                weather.append(weather_table['Condition'][i])
            else:
                weather.append(None)

        heavy_rain = False
        rain = False
        for condition in weather:
            if condition:
                for variation in ['Thunder', 'T-Storm', 'Storm', 'Lightning', 'Hail']:
                    if variation in condition:
                        return 'lightning or hail'

                if 'Heavy' in condition:
                    heavy_rain = True
                elif 'Rain' in condition:
                    rain = True

        if heavy_rain:
            return 'heavy rain'
        elif rain:
            return 'light rain'

        return 'fair'
    else:
        return 'error getting weather data'


def dst_status(date_str):
    """Returns string statuses depending on whether a day falls inside/outside/on the edge of dst.

    Parameters
    ----------
    date_str : str
        The date to be checked in yymmdd format.

    Returns
    -------
    str
        A status for the date: 'inside' if the date is inside dst, 'outside' if out, or 'beginning'/'end' for
        the boundaries.

    """

    year = int(params.CENTURY + date_str[0:2])
    month = int(date_str[2:4])
    day = int(date_str[4:])

    # January, February, and December are never DST
    if month < 3 or month > 11:
        return 'outside'
    # April to October are always DST
    elif 3 < month < 11:
        return 'inside'
    # DST starts on the second Sunday of March (which is always between the 8th and the 14th)
    elif month == 3:
        second_sunday = 8 + (6 - dt.datetime(year, month, 8).weekday())
        if day < second_sunday:
            return 'outside'
        elif day > second_sunday:
            return 'inside'
        else:
            return 'beginning'
    # DST ends on the first Sunday of November (so the previous Sunday must be before the 1st)
    else:
        first_sunday = 1 + (6 - dt.datetime(year, month, 1).weekday())
        if day < first_sunday:
            return 'inside'
        elif day > first_sunday:
            return 'outside'
        else:
            return 'end'


def dst_conversion(date_str, event_time, timezone_conversion):
    """Returns an updated UTC to local conversion number depending on the given date and time.

    Parameters
    ----------
    date_str : str
        The date to be converted in yymmdd format.
    event_time : float
        The time that the event occurred in units of seconds since the beginning of the day.
    timezone_conversion : int
        A number giving the hour difference between local time and UTC.

    Returns
    -------
    int
        An updated timezone conversion that accounts for dst.

    """

    temp_time = event_time + (timezone_conversion * params.SEC_PER_HOUR)
    if temp_time > params.SEC_PER_DAY:
        temp_time -= params.SEC_PER_DAY
        temp_date = roll_date_forward(date_str)
    elif temp_time < 0:
        temp_time += params.SEC_PER_DAY
        temp_date = roll_date_backward(date_str)
    else:
        temp_date = date_str

    temp_date_status = dst_status(temp_date)
    if temp_date_status == 'inside':  # Squarely inside dst
        return timezone_conversion + 1
    elif temp_date_status == 'outside':  # Squarely outside dst
        return timezone_conversion
    elif temp_date_status == 'beginning':  # Beginning of dst (2nd Sunday of March at 2:00AM)
        if temp_time >= params.TWO_AM:
            return timezone_conversion + 1
        else:
            return timezone_conversion
    else:  # End of dst (1st Sunday of November at 2:00AM)
        if (temp_time + params.SEC_PER_HOUR) >= params.TWO_AM:  # + sec_per_hour b/c temp time should be in dst
            return timezone_conversion
        else:
            return timezone_conversion + 1


def convert_clock_hour(clock_hour):
    """Converts a timestamp of the form hh:mm AM/PM into seconds since the beginning of the day."""

    meridiem = clock_hour.split()[1]
    hour = int(clock_hour.split()[0].split(':')[0])
    minute = int(clock_hour.split()[0].split(':')[1])

    # Converting from 12 hour time to 24 hour time
    if meridiem == 'AM' and hour == 12:  # midnight
        hour = 0
    elif meridiem == 'PM' and hour == 12:  # noon
        pass
    elif meridiem == 'PM':  # PM conversion
        hour += 12

    return float((hour * params.SEC_PER_HOUR) + (minute * 60))


def combine_data(detector):
    """Combines data from all scintillators into one set of arrays.

    Parameters
    ----------
    detector : tgfsearch.detectors.detector.Detector
        The Detector that data will be combined for.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Four numpy arrays:

            times
                An array containing the combined second-of-day times for multiple scintillators.
            energies
                An array containing the combined energies for multiple scintillators.
            count_scints
                An array containing scintillator names. Each entry corresponds to the scintillator
                that its corresponding count originated from.

    """

    times = []
    energies = []
    count_scints = []
    for scintillator in detector:
        if detector.data_present_in(scintillator):
            lm_frame = detector.get_attribute(scintillator, 'lm_frame', deepcopy=False)
            times.append(lm_frame['SecondsOfDay'])
            energies.append(lm_frame['energies'])
            count_scints.append(np.array([scintillator] * len(times[-1])))

    times = np.concatenate(times)
    energies = np.concatenate(energies)
    count_scints = np.concatenate(count_scints)

    sorting_order = np.argsort(times)
    times = times[sorting_order]
    energies = energies[sorting_order]
    count_scints = count_scints[sorting_order]
    return times, energies, count_scints


def separate_data(data, count_scints, start=None, stop=None):
    """Separates combined data from multiple scintillators into separate data for each scintillator.

    Parameters
    ----------
    data : numpy.ndarray
        An array of data to be separated.
    count_scints : numpy.ndarray
        An array containing scintillator names. Each entry corresponds to the scintillator
        that its corresponding data point originated from.
    start : int
        Optional. The beginning of the range to separate. If not provided, data will be separated starting at the
        beginning of the data array.
    stop : int
        Optional. The end of the range to separate. If not provided, data will be separated to the end of the
        data array.

    Returns
    -------
    dict
        A dictionary with separated data for each scintillator in count_scints.

    """

    if start is None:
        start = 0

    if stop is None:
        stop = len(data)

    data_dict = dict()
    # Recording all scintillators present
    count_scints_slice = count_scints[start:stop]
    for scintillator in count_scints_slice:
        if scintillator not in data_dict:
            data_dict[scintillator] = None

    # Separating the data into numpy arrays for each scintillator
    for scintillator in data_dict:
        data_dict[scintillator] = data[np.where(count_scints_slice == scintillator)[0] + start]

    return data_dict


def is_good_trace(trace):
    """Returns True if the given trace is likely to be interesting and False otherwise.

    Parameters
    ----------
    trace : pandas.core.frame.DataFrame
        A dataframe containing the trace data.

    Returns
    -------
    bool
        True if the given trace is likely to be interesting, False otherwise.

    """

    # Calculating the trace baseline using the first NUM_BINS_BASELINE bins of the trace
    if trace.size >= params.NUM_BINS_BASELINE:
        baseline_slice = trace['pulse'][0:params.NUM_BINS_BASELINE - 1]
    else:
        baseline_slice = trace['pulse']

    # If there is a mode, use it as the baseline. Otherwise, use the median
    pulse_mode = baseline_slice.mode()
    if len(pulse_mode) > 0:
        baseline = int(pulse_mode.iloc[0])
    else:
        baseline = int(baseline_slice.median())

    # Lower and upper bound for a count to be considered "in the vicinity" of the baseline
    lower_bound = baseline - params.TRIGGER_ABOVE_BASELINE
    upper_bound = baseline + params.TRIGGER_ABOVE_BASELINE

    above_baseline = 0
    below_baseline = 0
    for i in trace.index:
        value = trace['pulse'].iloc[i]
        # Adding up above and below baseline counts for the no saturation filter
        if value > upper_bound:
            above_baseline += 1
        elif value < lower_bound:
            below_baseline += 1

        # Getting to the first saturated count we can find and checking for a valid rising edge
        if value == 255:
            num_bins = 0
            # Working backwards from the saturated count
            for j in range(i - 1, i - params.LARGE_TRIGSPOT - 1, -1):
                # Loop will only traverse LARGE_TRIGSPOT bins or to the beginning of the trace, whichever is closer
                if j < 0:
                    break

                # Counting the number of bins that are above the baseline
                if trace['pulse'].iloc[j] > upper_bound:
                    num_bins += 1

            # Traces with valid rising edges (those that are gradual enough) pass
            if num_bins >= params.MIN_RISING_EDGE_BINS:
                return True
            else:
                return False

    # Traces that never reach saturation (and thus pass through the above loop unscathed) are checked further
    # Here we filter out traces that are mostly noise by checking the ratio of counts above the baseline to counts
    # below the baseline
    # Traces with no counts below the baseline are passed immediately
    if below_baseline == 0:
        return True

    # Noise spends about the same amount of time on either side, so if the trace is noisy this should be approximately 1
    ratio = above_baseline / below_baseline
    # Everything with a ratio that's suitably larger than 1 gets passed
    if ratio - 1 >= params.ABOVE_BASELINE_RATIO_THRESH:
        return True
    # Traces with a significant number of counts below the baseline (unusual) are also passed
    elif 1 - ratio >= params.BELOW_BASELINE_RATIO_THRESH:
        return True

    # Everything else is failed
    return False


def filter_traces(detector, scintillator):
    """Returns a list of traces that are likely to be interesting for the given scintillator.

    Parameters
    ----------
    detector : tgfsearch.detectors.detector.Detector
        The Detector containing the traces.
    scintillator : str
        The name of the scintillator of interest.

    Returns
    -------
    list
        A list containing the names of traces that are likely to be interesting for the requested scintillator.

    """

    good_traces = []
    trace_names = detector.get_trace_names(scintillator)
    for trace_name in trace_names:
        if is_good_trace(detector.get_trace(scintillator, trace_name, deepcopy=False)):
            good_traces.append(trace_name)

    return good_traces


def trace_to_counts(trace_energies):
    """Simulates the eMorpho response to trace data and returns the results."""

    # Adapted from esim_tools
    times = []
    energies = []
    peak = []
    tbnt = []
    psd = []

    n = len(trace_energies)
    di = int(params.DT / params.T_STEP)
    i = di
    baseline = np.median(trace_energies)
    thresh = params.TRACE_TRIGGER_THRESH / params.MV_PER_ADC
    while i < (n - params.DEADTIME_I - 1):
        # If we find a value above threshold
        if trace_energies[i] > (thresh + baseline):
            clip = trace_energies[i - 5:(i - 6 + params.INT_I)]

            energy = np.sum(clip - baseline)  # Integrate energy over INT_I samples starting with first > threshold
            partial = np.sum(clip[0:params.PARTIAL_INT_I] - baseline)  # Partial integration for PSD

            # Convert energy into channels to compare to real data spectrum
            norm_energy = energy * params.ENERGY_RESCALE
            norm_partial = partial * params.ENERGY_RESCALE

            times.append(i)
            energies.append(norm_energy)
            peak.append(np.max(clip))
            tbnt.append(len(np.where(clip < baseline - thresh)[0]))
            psd.append(norm_partial)

            i += params.DEADTIME_I
            # Paralyzable deadtime. Keep extending the window as long as the last sample of the
            # last interval is still high
            if params.DEADTIME_EXTEND > 0:
                while trace_energies[i - 1] > (thresh + baseline) and i < (n - di - params.DEADTIME_EXTEND):
                    i += params.DEADTIME_EXTEND

        else:
            i += 1

    return np.array(times), np.array(energies), np.array(peak), np.array(tbnt), np.array(psd)


def align_trace(trace, lm_frame, buff_no=0, trigspot=None):
    """Aligns the given trace with the given list mode data.

    Parameters
    ----------
    trace : pandas.core.frame.DataFrame
        A dataframe containing the trace data to be aligned.
    lm_frame : pandas.core.frame.DataFrame
        A dataframe containing the list mode data to be aligned against.
    buff_no : int
        Optional. The number of the trace buffer to be aligned (buffer zero by default).
    trigspot : int
        Optional. The location in the buffer where the trace was triggered (in units of samples).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Two numpy arrays. The first contains the aligned trace times in seconds of day,
        the second contains the bit-corrected trace pulse magnitude.

    """

    trace = trace[trace['BufferNo'] == buff_no]  # Only doing the alignment for the buffer we request
    if trace.size == 0:
        raise ValueError(f"no data for buff_no '{buff_no}'.")

    # Note: in the future, we might want to consider moving these bit shift operations to the data reader instead

    # Restoring four lost clock bits and converting to seconds
    trigger_time = trace['freeze'].iloc[0] * 16 * params.T_STEP
    trace_times = np.array(trace['Seconds']) + trigger_time
    # Restoring four lost pulse magnitude bits
    trace_energies = np.array(trace['pulse']) * 16

    # Simulating eMorpho response to trace data (effectively gives us a list mode analogue of the trace)
    sim_times = trace_to_counts(trace_energies)[0]
    if len(sim_times) < 2:
        raise ValueError('insufficient trace data to align.')

    sim_times = sim_times * params.T_STEP + trigger_time

    rollover_correction = params.ROLLOVER_PERIOD * params.T_STEP  # here because we're using this a decent amount

    # Converting wallclock to seconds and compensating for rollover
    wallclock_times = np.array(lm_frame['wc']) * params.T_STEP
    rollover_locs = np.where((wallclock_times - np.roll(wallclock_times, 1))[1:] < -800)[0]
    if len(rollover_locs) > 0:
        wallclock_times[(rollover_locs[0] + 1):] += rollover_correction

    # Compensating for rollover in the simulation times
    if sim_times[0] < wallclock_times[0]:
        sim_times += rollover_correction

    if sim_times[0] > wallclock_times[-1]:
        sim_times -= rollover_correction

    # Identifying the best alignment
    sim_times_us = sim_times * 1e6  # Trace times in microseconds
    wallclock_times_us = wallclock_times * 1e6  # List mode times in microseconds

    if trigspot is None:
        if buff_no == 0:  # Large buffer
            trigspot = params.LARGE_TRIGSPOT
        else:  # Smaller buffers
            trigspot = params.SMALL_TRIGSPOT

    # Making potential shift values (in microseconds)
    shifts = (np.arange(100) - 50) / 10 - trigspot * 12.5e-3

    # Getting the rough times of the event
    event_indices = np.where((wallclock_times_us > (sim_times_us.min() - 2)) &
                             (wallclock_times_us < (sim_times_us.max() + 2)))[0]
    if len(event_indices) == 0:
        raise ValueError('trace did not take place in given list mode data.')

    wallclock_times_us = wallclock_times_us[event_indices]

    best_score = 0
    best_shift = 0
    for shift in shifts:
        score = 0
        shifted_trace = sim_times_us + shift
        for val in shifted_trace:
            min_diff = np.absolute(val - wallclock_times_us).min()
            score += np.exp(-min_diff ** 5)

        if score > best_score:
            best_score = score
            best_shift = shift

    if best_score == 0:
        best_shift = np.median(shifts)

    # Returning the proper alignment
    event_start_wc = wallclock_times[event_indices[0]]
    if event_start_wc > rollover_correction:
        event_start_wc -= rollover_correction

    trace_times += best_shift * 1e-6 + lm_frame['SecondsOfDay'].iloc[event_indices[0]] - event_start_wc
    return trace_times, trace_energies


def channel_to_mev(energy_array, channels, energies):
    """Uses compton edges/photo peaks obtained from scintillator calibration to convert energy channels into MeV.

    Parameters
    ----------
    energy_array : numpy.ndarray
        An array containing energies for each count.
    channels : list
        A list containing the energy channels corresponding to the Compton edges/photo peaks.
    energies : list
        A list containing the energies of the Compton edges/photo peaks.

    Returns
    -------
    numpy.ndarray
        An array containing energies for each count in MeV.

    """

    a = (energies[0] - energies[1]) / (channels[0] - channels[1])
    b = energies[1] - a * channels[1]
    energy_array = a * energy_array + b
    return energy_array
