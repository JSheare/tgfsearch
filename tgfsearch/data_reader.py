"""A module containing functions for reading UCSC TGF group data files."""
import gzip as gzip
import json as json
import numpy as np
import pandas as pd
import re as re
import warnings as warnings
from datetime import datetime


def get_passtime():
    """Returns a fresh instance of the passtime dictionary needed for reading consecutive frames/files.

    Passtime entries:

    - lastsod: The second of day as calculated for the last event in the previous frame.
    - ppssod: The second of day as calculated for the last GPS pulse per second of the previous frame.
    - lastunix: Unix time (epoch seconds) for the last event of the previous frame (directly equivalent to lastsod
      regardless of data).
    - ppsunix: Unix time (epoch seconds) for the last GPS pulse per second of the previous frame (directly equivalent to
      ppssod, regardless of data).
    - lastwc: Bridgeport wall clock for the last event in the previous frame (no rollover corrections).
    - ppswc: Bridgeport wall clock for the last GPS pulse per second of the previous frame (no rollover corrections).
    - ppsage: the age (in frames) of the recorded pulse per second.
    - frlen: the length of the first frame read.
    - prevwc: The most recently parsed frame's wall clock column. Used to check for duplicate frame data.
    - started: flag for whether there is a previous frame. If 0, current passtime values will be ignored.

    Returns
    -------
    dict
        The passtime dictionary.

    """

    return {'lastsod': -1, 'ppssod': -1, 'lastunix': -1, 'ppsunix': -1, 'lastwc': -1, 'ppswc': -1, 'ppsage': -1,
            'frlen': -1, 'prevwc': None, 'started': 0}


def translate_flags(data, start=None, end=None, only_flags=None):
    """Translates the flags column of the given list mode dataframe into a dataframe with columns for each flag.

    Flag meanings:

    - pps: event was caused by an arriving GPS PPS (pulse per second) pulse.
    - or: event was out of ADC input range (computational over/under flow).
    - ov: event was out of MCA range, or had a negative value (computational over/under flow).
    - ps: event was flagged as piled up.
    - xt: event was caused by an external trigger.
    - he: high at end; i.e. ADC above trigger threshold at the end of the integration window.
    - hs: high at start; i.e. ADC above trigger threshold at the beginning of the integration window.
    - nr: ADC underflow; ADC hit negative rail (0 mV) sometime during integration window.
    - gps_sync: the GPS was in sync during the event.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        A pandas dataframe containing list mode data.
    start : int
        Optional. The start index of the slice to translate flags for. If not specified, an index of 0 (beginning of
        dataframe) will be used.
    end : int
        Optional. The end index of the slice to translate flags for. If not specified, an index of len(data) (end
        of dataframe) will be used.
    only_flags : list[str]
        Optional. A list of strings corresponding to flags to be translated; all other flags will be ignored. If left
        unspecified, all flags will be translated.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with columns of bools for each flag.

    """

    if 'flags' not in data.columns or 'file_type' not in data.attrs:
        raise ValueError('not a supported file type, or no flags column to translate.')

    if start is not None or end is not None:
        if start is None:
            start = 0

        if end is None:
            end = len(data.index)

        data_slice = data['flags'][start:end]
    else:
        data_slice = data['flags']

    # These lists are ordered from least to most significant bit
    if data.attrs['file_type'] == 'ssv_nrl_lm':
        columns = ['gps_sync', 'xt', 'pu', 'or', 'ov', 'pps']
    elif data.attrs['file_type'] == 'thor_lm':
        columns = ['gps_sync', 'nr', 'hs', 'he', 'xt', 'pu', 'ov', 'or', 'pps']
    elif data.attrs['file_type'] == 'json_nrl_lm':
        columns = ['gps_sync', 'xt', 'pu', 'ov', 'or', 'pps']
    else:
        raise ValueError('not a supported file type.')

    flags = pd.DataFrame()
    bit = 1
    for column in columns:
        if only_flags is None or (only_flags is not None and column in only_flags):
            flags[column] = ((data_slice & bit) / bit).astype(bool)

        bit *= 2

    # Reversing the column order to match the flag bit order
    columns = list(flags.columns)
    columns.reverse()
    flags = flags[columns]

    return flags


def read_files(file_file):
    """Reads a list of data files and returns them in a single dataframe. See read_file() for more information on column
    names.

    Parameters
    ----------
    file_file : str
        The name of the file containing the list of data files.

    Returns
    -------
        pandas.core.frame.DataFrame
            A pandas dataframe containing the data from the listed files.

    """

    with open(file_file, 'r') as file:
        lines = file.read().splitlines()

    file_frames = []
    passtime = get_passtime()
    for file in lines:
        print(f'Reading file: {file}')
        data, passtime = read_file(file, passtime)
        file_frames.append(data)

    return pd.concat(file_frames, ignore_index=True)


def read_file(file_name, passtime=None, killcr=False):
    """Reads the given data file and returns it as a pandas dataframe.

    List Mode Columns:

    - energies: the energy of each count (channel).
    - SecondsOfDay: the second of day of each count.
    - wc: the wallclock tick of each count.
    - flags (sometimes present): the flags for each count. See translate_flags() for an explanation of each flag.
    - peak (sometimes present): the high point of each count pulse (not baseline subtracted).
    - psd (sometimes present): the energy measured during each count's partial integration time.

    Trace Columns:

    - freeze: the wallclock tick when the trace trigger happened.
    - pulse: the waveform of the trace.
    - BufferNo: the buffer that the trace came from.
    - DateTime: the time when the trace was recorded.
    - Seconds: seconds relative to the beginning of the trace.

    Parameters
    ----------
    file_name : str
        The name of the data file to be read.
    passtime : dict
        Optional. A dictionary used to keep track of information between list mode frames. If not specified,
        one will be automatically instantiated with get_passtime().
    killcr: bool
        Optional. If True, maximum and low energy counts will be stripped out of list mode data. False by default.

    Returns
    -------
        tuple[pandas.core.frame.DataFrame, dict]
            A pandas dataframe containing the data and the passtime dictionary used to keep track of information
            between list mode frames.

    """

    if file_name[-2:] == 'gz':
        # Decompresses the file if it's compressed
        with gzip.open(file_name, 'rt') as file:
            lines = file.read().splitlines()
    else:
        with open(file_name, 'r') as file:
            lines = file.read().splitlines()

    if passtime is None:
        passtime = get_passtime()

    # This means that it's a trace file, so go to the trace reading function
    if 'xtr' in file_name:
        return _read_trace_file(lines), passtime

    # Only the json-formatted list mode files start with a 3 character (including a space) line
    if len(lines[0]) == 3:
        # Only the json-formatted NRL format has a named pps column
        if 'pps' in lines[5]:
            parser = _json_nrl_lm_parser(passtime, lines)
        else:
            parser = _thor_lm_parser(passtime, lines)
    else:
        # Early list mode files alternate (Time) - (Buffer) - (Time) - (Buffer); the newer version has 4 time tags
        # before a buffer. So, we can determine the type by testing whether the first line is a time tag
        if len(lines[2]) < 50:
            parser = _ssv_nrl_lm_parser(passtime, lines)
        else:
            parser = _godot_lm_parser(passtime, lines)

    return _read_lm_file(passtime, parser, killcr), passtime


def _ucsc_timestring_to_datetime(time_string):
    parts = time_string.split()
    parts[-1] = parts[-1].rjust(6, '0')
    return datetime.strptime(' '.join(parts), '%Y %m %d %H %M %S %f')


def _custom_warning_format(message, category, filename, lineno, line=None):
    return f'Warning: {message}\n'


# Raises a warning with a custom message
def _raise_warning(message):
    original = warnings.formatwarning
    try:
        # Setting a custom format for the warning string
        warnings.formatwarning = _custom_warning_format
        warnings.warn(message)
    finally:
        warnings.formatwarning = original


# A generator that parses godot-style list mode data files (previously mode 1)
def _godot_lm_parser(passtime, lines):
    """Format Info:
    - Every file contains a single blank line, a file creation timestamp line (?), and then all the frame blocks.
    - Each frame block consists of a frame data line and a timestamp line.
    - The timestamp line contains a timestamp corresponding to the time when the frame was requested by the computer (?).
    - The frame data line is just a collection of space-separated words representing the frame's data.
    - The first two words are the detector's serial number and the number of events in the frame.
    - The remaining words are the actual data.
    - Each event consists of three words. The first is the event energy, and the last two are 16-bit halves of the
        32-bit integer representing the event's wall clock tick.
    - The wall clock is either reset after every frame or rolls over after 2^32 ticks.
    - The ADC (theoretically) samples at a frequency of 8*10^7 hz, so this is the rate that the wall clock ticks at.
    """

    # Starting on line three (index two). Each frame is two lines long
    for i in range(2, len(lines), 2):
        frame_header = _ucsc_timestring_to_datetime(lines[i + 1])  # Header line is one ahead of the lm string line
        # Splitting the lm string into individual values. Throwing out the first two, which are unneeded
        frame_values = [int(x) for x in lines[i].split(' ')[2:]]
        # Making each row. Every row is three values long
        frame_data = pd.DataFrame(np.reshape(frame_values, (int(len(frame_values)/3), 3)))
        # Recreating full 32-bit wall clock values from two 16-bit words (one fine, one coarse)
        # 65536 is 2^16. Multiplying an int by it is the same as a 16 bit rightward shift
        wc = frame_data[1] + frame_data[2] * 65536
        energies = frame_data[0]
        frame_data = pd.concat([energies.rename('energies'), wc.rename('wc')], axis=1)
        # Eliminating duplicate data from the previous frame if it's present
        if passtime['prevwc'] is not None and frame_data['wc'][0] == passtime['prevwc'][0]:
            continue
            # Duplicate frames consist of the entire previous frame plus a small number of counts tacked on at
            # the end. The code below is meant to extract only the extra counts, but causes some minor time glitches
            # so we won't use it (for now)
            # frame_data.drop([j for j in range(0, len(passtime['prevwc']))], inplace=True)
            # frame_data.reset_index(inplace=True)

        # Before this date, the wallclock was always reset after each frame.
        if int(frame_header.strftime('%y%m%d')) < 200913:
            passtime['started'] = 0

        passtime['prevwc'] = frame_data['wc']

        frame_data.attrs['file_type'] = 'godot_lm'
        # Yielding: the frame's time header, the frame's data, the adc sampling rate, and the rollover period
        yield frame_header, frame_data, 8e7, 2**32


# A generator that parses ssv-formatted NRL-style list mode data files (previously mode 2)
def _ssv_nrl_lm_parser(passtime, lines):
    """Format Info:
    - Every file contains two lines of metadata followed by all the frame blocks.
    - Each frame block consists of six lines: a buffer line, four timestamp lines, and a frame data line.
    - The buffer line is just two numbers (separated by a space) that record which data buffer the frame was read from
        (first number) and whether that data buffer was full (second number).
    - The four timestamp lines record (in order):
        - When the frame was requested by the computer.
        - When the computer received the "full buffer" message.
        - When the computer asked for the frame to be sent.
        - When the frame finished arriving.
    - The frame data line is just a collection of space-separated words representing the frame's data.
    - The first nine words (of which only five are actually used) are (something), the detector's serial number,
        (something), the number of events in the frame, and the buffer number.
    - The remaining words are actual data.
    - Each event is six words long:
        - The event's PSD.
        - The event's energy.
        - One full word (16 bits) of the 51-bit integer representing the wall clock tick (fine).
        - Another full word of the wall clock tick integer (coarse).
        - The last full word of the wall clock tick integer (very coarse).
        - The last three bits of the wall clock tick integer (super coarse), and the event's flags.
    - The wall clock rolls over after 2^48 ticks.
    - The ADC (theoretically) samples at a frequency of 8*10^7 hz, so this is the rate that the wall clock ticks at.
    """

    rollover_period = 2**48
    # Starting on line eight (index seven). Each frame is six lines long
    for i in range(7, len(lines), 6):
        frame_header = _ucsc_timestring_to_datetime(lines[i - 4])  # Header line is four behind the lm string line
        # Splitting the lm string into individual words. Throwing out the first nine, which are unneeded
        frame_values = [int(x) for x in lines[i].split(' ')[9:]]
        # Making each row (event). Every row is six words long
        frame_data = pd.DataFrame(np.reshape(frame_values, (int(len(frame_values) / 6), 6)))
        energies = frame_data[1]
        # Recreating full 64-bit wall clock values from three 16-bit words (one fine, one coarse, and one very coarse)
        # and the extra three bits from the flags word
        # Casting to a 64-bit integer is necessary to prevent integer overflow
        # 65536 is 2^16. Multiplying an int by it is the same as a 16 bit rightward shift
        wc = (frame_data[2].astype('int64') +
              frame_data[3].astype('int64') * 65536 +
              frame_data[4].astype('int64') * 65536 ** 2 +
              (frame_data[5].astype('int64') & 7) * 65536 ** 3)
        # Format string converts x to its string representation as a binary integer with at least eight bits
        flags = frame_data[5]
        flags //= 8  # shifting everything right by three bits to get rid of the extra wall clock bits
        pps = (flags & 16) / 16  # The pps flag is in the most significant bit (16)
        flags *= 2  # shifting everything left by one bit to make room for the gps sync flag
        frame_data = pd.concat([energies.rename('energies'), wc.rename('wc'), pps.rename('pps'), flags.rename('flags')],
                               axis=1)
        # Eliminating duplicate data from the previous frame. This usually happens when buffers that are only partially
        # full are read out
        if passtime['prevwc'] is not None:
            # Data is duplicated from some midpoint to the end of the frame
            diff = np.where(frame_data['wc'].diff() < 0)[0]
            # Checking that this isn't a rollover, and dropping the duplicate rows if it isn't
            if len(diff) > 0 and frame_data['wc'][diff[0]] - frame_data['wc'][diff[0] - 1] < rollover_period / 4:
                frame_data.drop([j for j in range(diff[0], len(frame_data.index))], inplace=True)
                frame_data.reset_index(inplace=True)

            # The whole frame is duplicated data from before the previous frame
            if passtime['prevwc'][0] > frame_data['wc'][len(frame_data.index) - 1]:
                # Checking that this isn't a rollover, and skipping the whole frame if it isn't
                if passtime['prevwc'][0] - frame_data['wc'][len(frame_data.index) - 1] < rollover_period / 4:
                    continue

        passtime['prevwc'] = frame_data['wc']

        frame_data.attrs['file_type'] = 'ssv_nrl_lm'
        # Yielding: the frame's time header, the frame's data, the adc sampling rate, and the rollover period
        yield frame_header, frame_data, 8e7, rollover_period


# A generator that parses Thor-style list mode data files (previously mode 0)
def _thor_lm_parser(passtime, lines):
    """Format Info:
    - Every file is just a collection of frame blocks.
    - Each frame block consists of six lines: a buffer line, four timestamp lines, and a frame data line.
    - The buffer line is just two numbers (separated by a space) that record which data buffer the frame was read from
        (first number) and whether that data buffer was full (second number).
    - The four timestamp lines record (in order):
        - When the frame was requested by the computer.
        - When the computer received the "full buffer" message.
        - When the computer asked for the frame to be sent.
        - When the frame finished arriving.
    - The frame data line is a string containing all the frame's data.
    - After the detector's serial number and a space, the rest of the string is just the json-formatted data.
    - The wall clock rolls over after 2^36 ticks.
    - The ADC (theoretically) samples at a frequency of 8*10^7 hz, so this is the rate that the wall clock ticks at.
    """

    # Starting on line six (index five). Each frame is six lines long
    for i in range(5, len(lines), 6):
        frame_header = _ucsc_timestring_to_datetime(lines[i - 4])  # Header line is four behind the lm string line
        frame_data = pd.DataFrame(json.loads(re.sub('eRC[0-9]{4} ', '', lines[i]))['lm_data'])
        frame_data['pps'] = (frame_data['flags'] & 128) / 128  # The pps flag is in the most significant bit (128)
        frame_data['flags'] *= 2  # shifting everything left by one bit to make room for the gps sync flag
        frame_data.drop(columns=['num_events', 'psd', 'tbnt'], inplace=True)  # Dropping redundant columns
        frame_data.attrs['file_type'] = 'thor_lm'
        # Yielding: the frame's time header, the frame's data, the adc sampling rate, and the rollover period
        yield frame_header, frame_data, 8e7, 2**36


# A generator that parses json-formatted NRL-style list mode data files (previously mode 3)
def _json_nrl_lm_parser(passtime, lines):
    """Format Info:
    - Every file is just a collection of frame blocks.
    - Each frame block consists of six lines: a buffer line, four timestamp lines, and a frame data line.
    - The buffer line is just two numbers (separated by a space) that record which data buffer the frame was read from
        (first number) and whether that data buffer was full (second number).
    - The four timestamp lines record (in order):
        - When the frame was requested by the computer.
        - When the computer received the "full buffer" message.
        - When the computer asked for the frame to be sent.
        - When the frame finished arriving.
    - The frame data line is a string containing all the frame's data.
    - After the detector's serial number and a space, the rest of the string is just the json-formatted data.
    - Note: unlike Thor data, this format has a dedicated PPS (pulse per second) column built in.
    - The wall clock rolls over after 2^48 ticks.
    - The ADC (theoretically) samples at a frequency of 8*10^7 hz, so this is the rate that the wall clock ticks at.
    """

    # Starting on line six (index five). Each frame is six lines long
    for i in range(5, len(lines), 6):
        frame_header = _ucsc_timestring_to_datetime(lines[i - 4])  # Header line is four behind the lm string line
        frame_data = pd.DataFrame(json.loads(re.sub('eRC[0-9]{4} ', '', lines[i]))['lm_data'])
        # Setting up the flags column
        frame_data['flags'] = 0
        for column in ['pps', 'or', 'ov', 'pu', 'xt']:
            frame_data['flags'] += frame_data[column]
            # Shifting everything left by one bit.
            # Note: because of the way that this is structured, one bit of room will be implicitly left at the end,
            # which works out nicely because we need that space for the gps sync flag
            frame_data['flags'] *= 2

        frame_data.drop(columns=['num_events', 'xt', 'pu', 'ov', 'or'], inplace=True)  # Dropping redundant columns
        frame_data.attrs['file_type'] = 'json_nrl_lm'
        # Yielding: the frame's time header, the frame's data, the adc sampling rate, and the rollover period
        yield frame_header, frame_data, 8e7, 2**48


# Reads a list mode file and returns its data
def _read_lm_file(passtime, parser, killcr):
    data_list = []
    started = passtime['started']
    long_gap_counts = 25
    for header, data, adc_rate, rollover_period in parser:
        last_unix = passtime['lastunix']
        _process_data_timing(passtime, header, data, adc_rate, rollover_period)
        if started:
            dt = (data['SecondsOfDay'][0] + datetime(header.year, header.month, header.day).timestamp()) - last_unix
            if dt > long_gap_counts * (
                    (data['SecondsOfDay'][len(data.index) - 1] - data['SecondsOfDay'][0]) / len(data.index)):
                _raise_warning(f'long gap ({dt}s) between frames at {header}')

            if dt < 0:
                _raise_warning(f'anomalous clock backwards ({dt}s) at {header}')

        if killcr:
            data = data[data['energies'] < 65000]
            data = data[data['energies'] > 100]

        data_list.append(data)
        started = passtime['started']

        if 'pps' in data.columns:
            data.drop(columns='pps', inplace=True)

        if 'energy' in data.columns:
            data.rename(columns={'energy': 'energies'}, inplace=True)

    all_data = pd.concat(data_list, ignore_index=True)
    if 'index' in all_data.columns:
        all_data.drop(columns='index', inplace=True)

    # Saving some more memory by reducing the integer sizes of select columns
    all_data['energies'] = all_data['energies'].astype('int32')
    if 'flags' in all_data.columns:
        all_data['flags'] = all_data['flags'].astype('int16')

    if 'peak' in all_data.columns:
        all_data['peak'] = all_data['peak'].astype('int16')

    if 'psd' in all_data.columns:
        all_data['psd'] = all_data['psd'].astype('int32')

    return all_data


# Processes the timing information for a single frame of list mode data
def _process_data_timing(passtime, header, data, adc_rate, rollover_period):
    last_index = len(data.index) - 1

    # Every time we encounter the wallclock going significantly backwards (1/4 of range), assume that it's a rollover
    # Count how many times we've rolled over and multiply that by the rollover period to get each row's correction
    # This is about making the wc column monotonically increasing
    rollover_correction = (data['wc'].diff() < -rollover_period/4).cumsum() * rollover_period
    data['wc'] += rollover_correction

    # Get the header timestamp in unix time, and the unix time of the first second of the current day
    header_unix = header.timestamp()
    daystart_unix = datetime(header.year, header.month, header.day).timestamp()

    # Compensating for a missing frame right before the current one
    # If there's a gap of more than 1/4 the length of an average frame between the current frame and the last frame,
    # assume that the previous frame is missing
    if passtime['started']:
        first_count = header_unix - (data['wc'][last_index] - data['wc'][0]) / adc_rate
        if first_count - passtime['lastunix'] > passtime['frlen'] / 4:
            _raise_warning(f'previous frame may be missing at {header}')
            passtime['started'] = 0

    # Checking for a rollover between the last frame's PPS and the final count
    if (passtime['started'] and passtime['ppsage'] == 0 and
            passtime['lastwc'] - passtime['ppswc'] < -rollover_period / 4):
        passtime['lastwc'] += rollover_period

    # Checking for a rollover between the last and current frame
    if passtime['started'] and data['wc'][0] - passtime['lastwc'] < -rollover_period / 4:
        rollover_correction += rollover_period
        data['wc'] += rollover_period

    # Checking for missing pps at the end of the previous frame
    prev_missing_pps = False
    if passtime['started'] and passtime['lastsod'] - passtime['ppssod'] > 1:
        prev_missing_pps = True

    # Get PPS events (if available)
    if 'pps' in data:
        pps = np.where(data['pps'])[0]
        n_pps = len(pps)
    else:
        pps = None
        n_pps = 0

    # Start by basing everything on the header The last event in the frame is assumed to have happened at exactly the
    # header time. The wallclock rate is assumed to be precise
    if passtime['started']:
        # Second method: inter-frame, not using PPS. Interpolate all data in the frame using the time of the last
        # event of the previous frame as a starting point.
        unix_times = passtime['lastunix'] + (data['wc'] - passtime['lastwc']) / adc_rate
    else:
        # First method: intra-frame only. Interpolate all data in the frame according to the assumption that the last
        # event actually happened at header_unix seconds
        unix_times = header_unix - (data['wc'][last_index] - data['wc']) / adc_rate

    # If there aren't enough PPS points to do an intra-frame interpolation with, keep the header solution
    if n_pps < 2:
        if 'pps' in data:
            _raise_warning(f'frame with {n_pps} PPS at {header}')

    # Otherwise, use PPS to get a more accurate solution
    else:
        gps_sync = pd.Series([0 for i in range(len(unix_times))])
        drift_err = 1e4
        actual_rate = adc_rate  # Temporary value
        end_interpolated = False
        for i in range(n_pps - 1, -1, -1):
            # Inter-frame PPS interpolation (the beginning of the frame up to the first PPS)
            if i == 0:
                start = 0
                stop = pps[0]
                # Rounding the second pps in the pair (the interpolation basis) to the nearest second
                unix_times[stop] = round(unix_times[stop])
                if passtime['started'] and passtime['ppsage'] == 0:
                    # The number of wallclock ticks between two consecutive pps points
                    actual_rate = data['wc'][stop] - passtime['ppswc']
                # No prior frame to interpolate from. Just use the most recently calculated rate
                else:
                    unix_times[start:stop] = (unix_times[stop] -
                                              (data['wc'][stop] - data['wc'][start:stop]) / actual_rate)
                    break

            # Intra-frame PPS interpolation (everything else in the frame)
            else:
                start = pps[i - 1]
                stop = pps[i]
                # Rounding the second pps in the pair (the interpolation basis) to the nearest second
                unix_times[stop] = round(unix_times[stop])

                # The number of wallclock ticks between two consecutive pps points
                actual_rate = data['wc'][stop] - data['wc'][start]

            if start == stop:
                continue

            # Checking to see that we're in sync (actual wallclock tick rate is within a certain acceptable error)
            if abs(actual_rate - adc_rate) < drift_err:
                # start:stop + 1 will make gps sync true for the events between the PPS *and* the PPS points themselves
                gps_sync[start:stop + 1] = 1  # True
            # If not synced, try guessing that one PPS was skipped
            elif abs(actual_rate / 2 - adc_rate) < drift_err:
                if i == 0 and not prev_missing_pps:
                    _raise_warning(f'missing one PPS across frame boundary at {header}.')
                else:
                    _raise_warning(f'missing one PPS at {header}.')

                actual_rate /= 2
                # start:stop + 1 will make gps sync true for the events between the PPS *and* the PPS points themselves
                gps_sync[start:stop + 1] = 1  # True
            # Otherwise, out of sync
            else:
                if i == 0 and prev_missing_pps:
                    pass
                else:
                    _raise_warning(f'GPS sync failed at {header}.')

                actual_rate = adc_rate

            # Making unix time
            # Start + 1:stop will only update the times between the two pps points
            unix_times[start + 1:stop] = (unix_times[stop] -
                                          (data['wc'][stop] - data['wc'][start + 1:stop]) / actual_rate)

            # Extrapolating that events after the last PPS of the frame have the same actual rate as those right before
            if not end_interpolated:
                unix_times[stop + 1:] = (unix_times[stop] +
                                         (data['wc'][stop + 1:] - data['wc'][stop]) / actual_rate)
                # Checking for a missing pps (or several) between the last pps and the end of the frame
                end_diff = unix_times[last_index] - unix_times[stop]
                if end_diff > 1:
                    if end_diff > 2:
                        _raise_warning(f'GPS sync failed at {header}, end of frame.')
                    else:
                        _raise_warning(f'missing one PPS at {header}, end of frame.')

                end_interpolated = True

        # Storing gps sync info in the last flags bit
        data['flags'] += gps_sync

    data['SecondsOfDay'] = unix_times - daystart_unix
    # Checking to see if we've crossed a day boundary in this frame. When this happens, SecondsOfDay goes partially
    # negative. Counts in the current day become count - 86400
    negatives = np.where(data['SecondsOfDay'] < 0)[0]
    if len(negatives) > 0 and data['SecondsOfDay'].max() < 5000:
        # Updating data so that when the day boundary is crossed, SecondsOfDay will start over again from zero
        data.iloc[0:negatives[-1] + 1, data.columns.get_loc('SecondsOfDay')] += 86400

    # Undoing rollover correction
    data['wc'] -= rollover_correction

    # Updating passtime
    passtime['lastunix'] = unix_times[last_index]
    passtime['lastsod'] = data['SecondsOfDay'][last_index]
    passtime['lastwc'] = data['wc'][last_index]
    if n_pps > 0:
        passtime['ppswc'] = data['wc'][pps[-1]]
        passtime['ppsunix'] = unix_times[pps[-1]]
        passtime['ppssod'] = data['SecondsOfDay'][pps[-1]]
        passtime['ppsage'] = 0
    elif passtime['ppsage'] >= 0:
        passtime['ppsage'] += 1

    # Recording the length of the very first frame read. Used to check for prior missing frames later
    if not passtime['started']:
        passtime['frlen'] = data['SecondsOfDay'][last_index] - data['SecondsOfDay'][0]
        passtime['started'] = 1


# Reads a trace file and returns its data
def _read_trace_file(lines):
    data_list = []
    for i in range(4, len(lines), 5):
        data = pd.DataFrame.from_dict(json.loads(re.sub("eRC[0-9]{4} [0-9]", "", lines[i])))
        # There should be some significance to how this affects the time relationships?
        data['BufferNo'] = int(lines[i][8:9])
        data['DateTime'] = datetime.strptime(lines[i - 1], "%Y %m %d %H %M %S %f")
        data['Seconds'] = [x * 1.25e-8 for x in range(len(data))]
        data_list.append(data)

    trace = pd.concat(data_list)
    trace.attrs['file_type'] = 'trace'
    return trace
