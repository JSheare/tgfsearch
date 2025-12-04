"""Tools for use by the TGF search program and its modules."""
import datetime as dt
import io as io
import numpy as np
import os as os
import pandas as pd
import pickle as pickle
import struct as struct
import zoneinfo as zi
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
            to the Detector's export directory.

    Returns
    -------
    str
        The path to the pickle file (including its name).

    """

    if path is None:
        path = f'{detector.get_export_loc()}'

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


def get_weather_table(local_date, deployment_info):
    """Scrapes weather data from the internet and returns the results as a pandas data frame.

    Parameters
    ----------
    local_date : str
        The local date that weather data is being requested for in yyyy-mm-dd format.
    deployment_info : dict
        A dictionary containing deployment information, including the timezone identifier and weather station callsign.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with weather information for the given day. Time entries are epoch timestamps. An empty
        table is returned if the scraping fails.

    """

    local_dt = dt.datetime(int(local_date[0:4]), int(local_date[5:7]), int(local_date[8:10]),
                           tzinfo=zi.ZoneInfo(deployment_info['tz_identifier']))
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Runs chrome in headless mode (no browser tab)
        # The below options prevent an annoying logging entry from being printed to stdout
        chrome_options.add_experimental_option("excludeSwitches", ['enable-logging'])
        chrome_options.add_argument("--log-level=3")
        chrome_options.set_capability("browserVersion", "117")
        chrome_options.add_argument("start-maximized")

        driver = webdriver.Chrome(options=chrome_options)

        url = (f'https://www.wunderground.com/history/daily/'
                   f'{deployment_info["weather_station"]}/date/{local_dt.date()}')

        driver.get(url)
        tables = WebDriverWait(driver, 20).until(ec.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
        table = pd.read_html(io.StringIO(tables[1].get_attribute('outerHTML')))[0].dropna()
    except:
        return pd.DataFrame()

    local_daystart_timestamp = (local_dt.timestamp() - (local_dt.hour * params.SEC_PER_HOUR) - (local_dt.minute * 60) -
                                local_dt.second - (local_dt.microsecond * 1e-6))
    table['Time'] = [local_daystart_timestamp + convert_clock_hour(hour) for hour in table['Time']]
    return table


def assemble_weather_info(detector, event_time, weather_cache):
    """Returns a table of weather information from around the vicinity of the given event time. The function will
    attempt to make a table with at least +/- params.WEATHER_PADDING hours around the given time. An empty table will be
    returned if no information could be retrieved."""
    event_timestamp = event_time + detector.first_sec
    local_dt = dt.datetime.fromtimestamp(event_timestamp).astimezone(zi.ZoneInfo(detector.deployment['tz_identifier']))
    local_date = local_dt.date()
    local_date_str = str(local_date)
    if local_date_str in weather_cache:
        table = weather_cache[local_date_str]
    else:
        table = get_weather_table(local_date_str, detector.deployment)
        # Returning an empty table immediately to avoid erroneously caching it and/or concatenating with it
        if table.empty:
            return table

        weather_cache[local_date_str] = table

    # Checking to see if we need to retrieve weather data from the previous local day based on the window size
    left_dt = dt.datetime.fromtimestamp(
        event_timestamp - params.WEATHER_PADDING * params.SEC_PER_HOUR).astimezone(
        zi.ZoneInfo(detector.deployment['tz_identifier']))
    left_date = left_dt.date()
    left_date_str = str(left_date)
    if left_date < local_date:
        if left_date_str not in weather_cache:
            left_table = get_weather_table(left_date_str, detector.deployment)
            # If this table is empty, concatenation will fail. To avoid this, and to avoid caching the empty table, we
            # just return the table that we already have
            if left_table.empty:
                return table

            weather_cache[left_date_str] = left_table

        return pd.concat([weather_cache[left_date_str], table])

    # Checking to see if we need to retrieve weather data from the next local day based on the window size
    right_dt = dt.datetime.fromtimestamp(
        event_timestamp + params.WEATHER_PADDING * params.SEC_PER_HOUR).astimezone(
        zi.ZoneInfo(detector.deployment['tz_identifier']))
    right_date = right_dt.date()
    right_date_str = str(right_date)
    if right_date > local_date:
        if right_date_str not in weather_cache:
            right_table = get_weather_table(right_date_str, detector.deployment)
            # If this table is empty, concatenation will fail. To avoid this, and to avoid caching the empty table, we
            # just return the table that we already have
            if right_table.empty:
                return table

            weather_cache[right_date_str] = right_table

        return pd.concat([table, weather_cache[right_date_str]])

    return table


def get_weather_conditions(detector, event_time, weather_cache=None):
    """Scrapes weather underground and returns the weather around the time of an event.

    Parameters
    ----------
    detector : tgfsearch.detectors.detector.Detector
        The Detector that contains the name of the nearest weather station.
    event_time : float
        The time that the event occurred at during the day in units of seconds since beginning of day.
    weather_cache : dict
       Optional. A cache containing weather tables that have already been retrieved. The keys are local dates in
       yyyy-mm-dd format.

    Returns
    -------
    str
        The weather conditions around the time of the event as a string.

    """

    weather_table = assemble_weather_info(detector, event_time, weather_cache)
    if not weather_table.empty:
        # Finds the time in the table that's closest to the time of the event
        event_timestamp = event_time + detector.first_sec
        index = 0
        best_diff = float('inf')
        best_index = 0
        for timestamp in weather_table['Time']:
            diff = abs(event_timestamp - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_index = index

            index += 1

        # Gets the weather conditions at the closest hour to the event and the surrounding params.WEATHER_PADDING hours
        weather = []
        for i in range(best_index - params.WEATHER_PADDING, best_index + params.WEATHER_PADDING + 1):
            if 0 <= i < index:
                weather.append(weather_table['Condition'][i])

        heavy_rain = False
        rain = False
        for condition in weather:
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
