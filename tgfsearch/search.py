"""A script that searches for TGFs and glows."""
import datetime as dt
import gc as gc
import glob as glob
import heapq
import json as json
import matplotlib as matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os as os
import psutil as psutil
import scipy as sp
import sys as sys
import traceback as traceback
import warnings as warnings

# Adds parent directory to sys.path. Necessary to make the imports below work when running this file as a script
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import tgfsearch.parameters as params
import tgfsearch.tools as tl
from tgfsearch.detectors.adaptive_detector import AdaptiveDetector
from tgfsearch.detectors.detector import Detector
from tgfsearch.helpers.long_event import LongEvent
from tgfsearch.helpers.short_event import ShortEvent
from tgfsearch.helpers.trace_info import TraceInfo


# Returns True if the provided unit is a valid detector name and False otherwise
def is_valid_detector(unit):
    unit_upper = unit.upper()
    if unit_upper == 'ADAPTIVE':
        return True
    else:
        try:
            with open(f'{os.path.dirname(os.path.realpath(__file__))}/config/detector_config.json', 'r') as file:
                identities = json.load(file)['identities']

            return unit_upper in identities
        except json.decoder.JSONDecodeError:
            raise SyntaxError('invalid syntax in detector config file.')


# Checks whether a search with the given dates and detector is a valid one
def is_valid_search(first_date, second_date, unit, print_feedback=False):
    # Checks that both dates are digits in the proper format
    if not first_date.isdigit() or not second_date.isdigit() \
            or len(first_date) != 6 or len(second_date) != 6:
        if print_feedback:
            print('Error: not a valid date. BOTH dates must be in yymmdd format.')

        return False

    # Checks that both dates are sequential
    if int(first_date) > int(second_date):
        if print_feedback:
            print('Error: second date must be AFTER first date.')

        return False

    # Checks that a valid detector has been entered
    if unit == '':
        if print_feedback:
            print('Error: no detector specified.')

        return False
    elif not is_valid_detector(unit):
        if print_feedback:
            print('Error: not a valid detector.')

        return False

    return True


# Returns the correct detector object based on the parameters provided
def get_detector(unit, date_str):
    if unit.upper() == 'ADAPTIVE':
        return AdaptiveDetector(date_str)
    else:
        return Detector(unit, date_str)


# Makes the modes dict used by many of the program's functions
def get_modes(mode_info):
    modes = dict()
    # Aircraft mode
    modes['aircraft'] = True if '--aircraft' in mode_info else False

    # All scintillators mode (all the scintillators will be checked individually by the short event search algorithm)
    modes['allscints'] = True if '--allscints' in mode_info else False

    # Clean energy mode (strip out max energy and low energies from the data)
    modes['clnenrg'] = True if '--clnenrg' in mode_info else False

    # Custom mode (use custom import and/or export directories
    modes['custom'] = True if '-c' in mode_info else False

    # Onescint mode (only the highest priority scintillator will be checked by the short event search algorithm)
    modes['onescint'] = True if '--onescint' in mode_info else False

    # Pickle mode
    modes['pickle'] = True if '--pickle' in mode_info else False

    # Modes for skipping over certain algorithms (mostly to speed up testing)
    modes['sktrace'] = True if '--sktrace' in mode_info else False  # Skip trace filtering
    modes['skshort'] = True if '--skshort' in mode_info else False  # Skip short event search
    modes['skglow'] = True if '--skglow' in mode_info else False  # SKip long event search

    # Allscints includes onescint, so disable onescint if they're both present
    if modes['allscints'] and modes['onescint']:
        modes['onescint'] = False

    return modes


# Returns the maximum allowed fraction of available system memory that the program is allowed to use for data
def get_max_mem_frac():
    available_memory = psutil.virtual_memory()[1]
    if available_memory * params.MEMORY_ALLOWANCE_FRAC < params.ABS_MEMORY_ALLOWANCE:
        return params.MEMORY_ALLOWANCE_FRAC

    return params.ABS_MEMORY_ALLOWANCE / available_memory


# Logs errors that occur during daily searches
def log_error(detector, modes, ex):
    tl.print_logger('\n', detector.log)
    tl.print_logger(f'Search could not be completed due to the following error: {ex}', detector.log)
    tl.print_logger('See error log for details.', detector.log)
    with open(f'{detector.get_export_loc()}/err.txt', 'w') as err_file:
        print('Info:', file=err_file)
        print(f'{detector.date_str} {detector.unit}', file=err_file)
        for mode in modes:
            print(f'{mode}: {modes[mode]}', file=err_file)

        print('', file=err_file)
        err_file.write(traceback.format_exc())


# Plots the given list of traces
def plot_traces(detector, scintillator, trace_names):
    if trace_names:
        # Makes the trace plot path
        plot_path = f'{detector.get_export_loc()}/traces'
        tl.make_path(plot_path)
        for trace_name in trace_names:
            trace = detector.get_trace(scintillator, trace_name, deepcopy=False)
            trace_single_buff = trace[trace['BufferNo'] == trace['BufferNo'].iloc[0]]
            plot_name = 'eRC' + trace_name.split('eRC')[-1][1:].split('.')[0]

            # Makes the figure
            figure = plt.figure()
            ax = figure.add_subplot()
            figure.suptitle(plot_name)
            ax.plot(trace_single_buff['Seconds'], trace_single_buff['pulse'])
            ax.set_xlabel('Seconds')
            ax.set_ylabel('Pulse Magnitude')

            # Saves the figure
            figure.savefig(f'{plot_path}/{plot_name}.png')
            figure.clf()
            plt.close(figure)
            gc.collect()


# Searches through all traces in detector and returns lists of good ones for each scintillator. Also plots good traces
def find_traces(detector):
    # Filtering traces and setting up trace_dict, which keeps track of filtered trace names
    trace_dict = {}
    for scintillator in detector:
        trace_dict[scintillator] = tl.filter_traces(detector, scintillator)

        # Making plots for the traces that passed through the filter
        plot_traces(detector, scintillator, trace_dict[scintillator])

    return trace_dict


# Returns the high/low channel ratio for the given range
def get_hl_channel_ratio(energies, start, stop):
    assert params.CHANNEL_SEPARATION >= 0
    low_channel_start = params.LOW_CHANNEL_START
    low_channel_end = low_channel_start + params.CHANNEL_RANGE_WIDTH
    high_channel_start = params.LOW_CHANNEL_START + params.CHANNEL_RANGE_WIDTH + params.CHANNEL_SEPARATION
    high_channel_end = params.LOW_CHANNEL_START + 2 * params.CHANNEL_RANGE_WIDTH + params.CHANNEL_SEPARATION
    low_channel_counts = 0
    high_channel_counts = 0
    for i in range(start, stop):
        count_energy = energies[i]
        if low_channel_start <= count_energy <= low_channel_end:
            low_channel_counts += 1
        elif high_channel_start <= count_energy <= high_channel_end:
            high_channel_counts += 1

    if low_channel_counts > 0:
        return high_channel_counts / low_channel_counts

    return float('inf')


# Returns the n highest energy count on the given range
def get_n_highest_count(energies, start, stop, n):
    if n <= 0 or n > stop - start:
        return 0.

    priority_queue = []
    for i in range(start, stop):
        if len(priority_queue) < n:
            heapq.heappush(priority_queue, energies[i])
        else:
            heapq.heappushpop(priority_queue, energies[i])

    return heapq.heappop(priority_queue)


# Returns the clumpiness and high energy lead values for the given range
def get_clumpiness_and_hel(times, energies, start, stop):
    length = stop - start
    if length == 0:
        return 0., 0.

    clumpiness = 0
    clump_counts = 0
    high_energy_lead = 0
    leading_counts = 0
    prev_time = times[start]
    for i in range(start + 1, stop):
        count_time = times[i]
        difference = count_time - prev_time
        prev_time = count_time  # Numpy array indexing is slow, so I'd rather do it as little as possible
        if difference < params.DIFFERENCE_THRESH:
            clump_counts += 1
            if clump_counts == 1:
                leading_counts += 1
                # i - 1 because the clump starts on the previous index
                if energies[i - 1] >= params.HIGH_ENERGY_LEAD_THRESH:
                    high_energy_lead += 1

            # For the last count in the event
            if i == stop - 1:
                clump_counts += 1

        else:
            # Adding to the clumpiness when there's a clump of three or more
            if clump_counts >= 3:
                clumpiness += 1
                clump_counts = 0

            # Adding to the clumpiness when the gap between sufficient counts is greater than the threshold
            if difference >= params.GAP_THRESH and clump_counts == 0:
                clumpiness += 1

            clump_counts = 0

    clumpiness /= length
    if leading_counts > 0:
        high_energy_lead /= leading_counts

    return clumpiness, high_energy_lead


# Short event search algorithm
def short_event_search(detector, modes, scintillator, rollgap, times, energies):
    # Stats
    total_potential_events = 0
    total_threshold_reached = 0
    removed_len = 0
    removed_channel_ratio = 0
    removed_low_energy = 0
    removed_crs = 0

    # Checks for an event by looking for a certain number of counts (rollgap + 1) in a small timeframe
    short_events = []
    event_start = 0
    event_length = 0
    for i in range(len(times)):
        # Note: we're doing it with indexing like this to save memory. It's slower (numpy array indexing is about 5x
        # slower than normal list indexing due to overhead), but it also uses significantly less memory because
        # we aren't making a copy of the data array with a call to np.roll
        rolled_index = i + rollgap if i + rollgap < len(times) else (i + rollgap) - len(times)
        interval = times[rolled_index] - times[i]
        if 0 < interval <= params.SHORT_EVENT_TIME_SPACING:
            # Records the beginning index of a potential event
            if event_length == 0:
                event_start = i
                event_length = 1 + rollgap  # 1 for first count, rollgap for the others
                total_potential_events += 1
            # Measures the length of a potential event
            else:
                event_length += 1

            # Counts the total number of times that the detection threshold was reached
            total_threshold_reached += 1

        # Records the rough length of a potential event
        elif interval > params.SHORT_EVENT_TIME_SPACING and event_length > 0:
            # Runs potential event through filters
            event_stop = event_start + event_length
            good_event = True

            # Checks that the event is above the minimum length
            if event_length < params.SHORT_EVENT_MIN_COUNTS:
                good_event = False
                removed_len += 1

            # Checks that there are fewer counts in the higher energy channels than the low energy channels
            # High/low channel ratio is not checked for THOR or for events with < GOOD_LEN_THRESH counts
            if (good_event and event_length >= params.GOOD_LEN_THRESH and detector.is_named('GODOT') and
                    get_hl_channel_ratio(energies, event_start, event_stop) > params.CHANNEL_RATIO):
                good_event = False
                removed_channel_ratio += 1

            # Eliminates the low energy events, which are likely just noise
            # At least min_noise_counts must be above noise_cutoff_energy
            if (good_event and get_n_highest_count(energies, event_start, event_stop, params.MIN_NOISE_COUNTS)
                    < params.NOISE_CUTOFF_ENERGY):
                good_event = False
                removed_low_energy += 1

            # Eliminates events that are just successive cosmic ray showers
            if (good_event and modes['aircraft'] and event_length < params.GOOD_LEN_THRESH and
                    get_clumpiness_and_hel(times, energies, event_start, event_stop)[0] >= params.CLUMPINESS_THRESH):
                good_event = False
                removed_crs += 1

            # Adds the event to the list if it passed all the filters
            if good_event:

                short_events.append(ShortEvent(event_start, event_length, scintillator))

            event_start = 0
            event_length = 0

    tl.print_logger(f'{len(short_events)} potential events recorded', detector.log)
    if removed_len > 0:
        tl.print_logger(f'{removed_len} events removed due to insufficient length', detector.log)

    if removed_channel_ratio > 0:
        tl.print_logger(f'{removed_channel_ratio} events removed due to noise '
                        f'(bad high-to-low channel ratio)', detector.log)

    if removed_low_energy > 0:
        tl.print_logger(f'{removed_low_energy} events removed due to noise '
                        f'(minimum energy threshold not reached)', detector.log)

    if removed_crs > 0:
        tl.print_logger(f'{removed_crs} events removed due to successive CRS', detector.log)

    tl.print_logger(f'Detection threshold reached {total_threshold_reached} times', detector.log)

    return short_events


# Calculates and records subscores and final score for the given short event
def calculate_se_score(detector, event, weather_cache, times, energies):
    # Calculating and recording the length subscore
    event.len_subscore = event.length / params.GOOD_LEN_THRESH

    clumpiness, high_energy_lead = get_clumpiness_and_hel(times, energies, event.start, event.stop)
    # Calculating and recording the clumpiness subscore
    event.clumpiness_subscore = 1 / (1 + np.e ** (40 * (clumpiness - params.CLUMPINESS_TOSSUP)))

    # Calculating and recording the high energy leading count subscore
    event.hel_subscore = np.e ** -(5.3 * high_energy_lead)

    # Calculating the recording the weather subscore
    if detector.deployment['weather_station'] != '':
        event.weather_conditions = tl.get_weather_conditions(detector, times[event.start], weather_cache=weather_cache)
        match event.weather_conditions:
            case 'lightning or hail':
                event.weather_subscore = 1
            case 'heavy rain':
                event.weather_subscore = 0.75
            case 'light rain':
                event.weather_subscore = 0.5
            case 'fair':
                event.weather_subscore = 0
            case _:
                event.weather_subscore = -1
    else:
        event.weather_subscore = -1

    # Calculating and recording the final score
    assert (params.LEN_WEIGHT + params.CLUMPINESS_WEIGHT + params.HEL_WEIGHT + params.WEATHER_WEIGHT) == 1.0
    # If weather info couldn't be obtained, the weather subscore is removed and the remaining weights are adjusted
    # so that they stay proportional to one another
    if event.weather_subscore == -1:
        proportionality = 1 / (params.LEN_WEIGHT + params.CLUMPINESS_WEIGHT + params.HEL_WEIGHT)
        event.total_score = proportionality * (params.LEN_WEIGHT * event.len_subscore +
                                               params.CLUMPINESS_WEIGHT * event.clumpiness_subscore +
                                               params.HEL_WEIGHT * event.hel_subscore)
    else:
        event.total_score = (params.LEN_WEIGHT * event.len_subscore +
                             params.CLUMPINESS_WEIGHT * event.clumpiness_subscore +
                             params.HEL_WEIGHT * event.hel_subscore +
                             params.WEATHER_WEIGHT * event.weather_subscore)


# Finds the list mode file(s) associated with a short event
def find_se_files(detector, event, times, count_scints):
    if count_scints is None:
        scintillators_left = 1
    else:
        scintillators_left = 0
        for scintillator in detector:
            if detector.data_present_in(scintillator):
                scintillators_left += 1

    for i in range(event.start, event.stop):
        if count_scints is not None:
            scintillator = count_scints[i]
        else:
            scintillator = event.scintillator

        if scintillator not in event.lm_files:
            event_time = times[i]
            event.lm_files[scintillator] = detector.find_lm_file(scintillator, event_time)
            scintillators_left -= 1

        # Terminates the function early if we've already found files for the max number of scintillators possible
        if scintillators_left == 0:
            break


# Finds the traces associated with a short event
def find_se_traces(detector, event, trace_dict, times, count_scints):
    if count_scints is None:
        scintillators_left = 1
    else:
        scintillators_left = 0
        for scintillator in detector:
            if detector.data_present_in(scintillator):
                scintillators_left += 1

    for i in range(event.start, event.stop):
        if count_scints is not None:
            scintillator = count_scints[i]
        else:
            scintillator = event.scintillator

        if scintillator not in event.traces:
            event_time = times[i]
            event_lm_file_data = detector.get_lm_file(scintillator, detector.find_lm_file(scintillator, event_time))
            potential_matches = detector.find_matching_traces(scintillator, event_time,
                                                              trace_list=trace_dict[scintillator])
            trace_info = None
            best_diff = float('inf')
            for trace_name in potential_matches:
                trace = detector.get_trace(scintillator, trace_name)
                buff_no = trace['BufferNo'].iloc[0]
                # Aligning the trace and checking how close it is to the event
                try:  # In case trace isn't in the same file after all
                    trace_times, trace_energies = tl.align_trace(trace, event_lm_file_data, buff_no)
                    diff = abs(trace_times[0] - event_time)
                    # The best match is whichever trace is closest to the event
                    if diff < best_diff:
                        best_diff = diff
                        trace_info = TraceInfo(trace_name, buff_no, trace_times, trace_energies)

                except ValueError:
                    continue

            if (trace_info is not None and len(np.where((trace_info.times >= times[event.start]) &
                                                        (trace_info.times <= times[event.stop]))[0]) > 0):
                event.traces[scintillator] = trace_info
                scintillators_left -= 1

            # Terminates the function early if we've already found files for the max number of scintillators possible
            if scintillators_left == 0:
                break


# Makes the scatter plot for a short event
def make_se_scatterplot(detector, event, times, energies, count_scints):
    timestamp = dt.datetime.fromtimestamp(times[event.start] + detector.first_sec, dt.UTC)
    # Subplot timescales
    timescales = [params.SE_TIMESCALE_ONE, params.SE_TIMESCALE_TWO, params.SE_TIMESCALE_THREE]

    # Dot colors. Dots for unsupported scintillators will be black (see below)
    try:
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/config/search_config.json', 'r') as file:
            colors = json.load(file)['short_event_search_colors']
    except json.decoder.JSONDecodeError:
        raise SyntaxError('invalid syntax in search config file.')

    # Determining the best place to center the plot around and slicing the arrays accordingly
    event_times = times[event.start:event.stop]
    best_index = np.argmin(np.abs(event_times - np.roll(event_times, 1))) + event.start
    half_interval = params.SE_TIMESCALE_THREE / 2
    left_edge = best_index
    right_edge = best_index

    # Finding the best left edge to use
    while True:
        if left_edge < 0 or times[best_index] - times[left_edge] > half_interval:
            left_edge += 1
            break

        left_edge -= 1

    # Finding the best right edge to use
    while True:
        if right_edge >= len(times) or times[right_edge] - times[best_index] > half_interval:
            right_edge -= 1
            break

        right_edge += 1

    if count_scints is not None:
        times_dict = tl.separate_data(times, count_scints, left_edge, right_edge)
        energies_dict = tl.separate_data(energies, count_scints, left_edge, right_edge)
    else:
        times_dict = {event.scintillator: times[left_edge:right_edge]}
        energies_dict = {event.scintillator: energies[left_edge:right_edge]}

    figure = plt.figure(figsize=[20, 11.0], dpi=150.)
    figure.suptitle(f'{event.scintillator} Event {str(event.number)}, ' 
                    f'{timestamp} UTC, ' 
                    f'{event.length} counts \n Weather: {event.weather_conditions} \n'
                    f'Score: {"%.3f" % event.total_score}, Rank: {event.rank}', fontsize=20)
    ax1 = figure.add_subplot(3, 1, 1)
    ax2 = figure.add_subplot(3, 1, 2)
    ax3 = figure.add_subplot(3, 1, 3)
    ax_list = [ax1, ax2, ax3]
    assert len(ax_list) == len(timescales)

    best_time = event_times[best_index - event.start]
    for i in range(len(ax_list)):
        ts = timescales[i]
        ax = ax_list[i]
        ax.set_xlim(xmin=best_time - (ts / 2), xmax=best_time + (ts / 2))

        dot_size = 5 if ts == params.SE_TIMESCALE_ONE else 3  # makes larger dots for top plot
        ax.set_yscale('log')
        ax.set_ylim((0.6, 1e5))
        for scintillator in times_dict:
            color = colors[scintillator] if scintillator in colors else 'k'
            # 0.6 to avoid annoying divide by zero warnings
            ax.scatter(times_dict[scintillator], energies_dict[scintillator] + 0.6,
                       s=dot_size, zorder=1, alpha=params.DOT_ALPHA, label=scintillator, color=color)

            # Plotting traces on only the first subplot
            if ts == params.SE_TIMESCALE_ONE and scintillator in event.traces:
                # Dividing by 100 makes the trace appear a little lower in the plot, which looks better
                ax.plot(event.traces[scintillator].times, (event.traces[scintillator].energies + 0.6) / 100,
                        zorder=-1, alpha=params.DOT_ALPHA, color=color)

        ax.set_xlabel(f'Time (Seconds, {ts}s total)')
        ax.set_ylabel('Energy Channel')
        # Lines appear (100*percent)% to the left or right of event start/stop depending on subplot timescale
        percent = 0.001
        # Event start line is orange, end line is red
        ax.vlines([event_times[0] - percent * ts, event_times[-1] + percent * ts], 0, 1e5,
                  colors=['orange', 'r'], linewidth=1, zorder=-1, alpha=0.3)

    # Adds a legend to the plot
    if count_scints is not None:
        ax3.legend(loc='lower right')

    # Adds the name of the relevant list mode file (and trace, if applicable) to the scatter plot
    if count_scints is None:
        figure.text(0.5, 0.03, f'Obtained from {event.lm_files[event.scintillator]}'
                    + (f'\nand {event.traces[event.scintillator].trace_name}' if
                       event.scintillator in event.traces else ''),
                    fontsize=15, horizontalalignment='center')

    # Saves the scatter plot
    # Note: with this code, if an event happens in that 200-300 seconds of the next day that are included in the
    # last file, the image will have the wrong date in its name (though the timestamp in the scatter plot title will
    # always be correct)
    scatter_path = f'{detector.get_export_loc()}/scatter_plots'
    tl.make_path(scatter_path)
    event_num_padding = '0' * (len(str(params.MAX_PLOTS_PER_SCINT)) - len(str(event.number)))
    rank_padding = '0' * (len(str(params.MAX_PLOTS_PER_SCINT)) - len(str(event.rank)))
    figure.savefig(f'{scatter_path}/'
                   f'{timestamp.strftime("%y%m%d_%H%M%S")}_'
                   f'{event.scintillator}_'
                   f'event{event_num_padding}{event.number}_'
                   f'rank{rank_padding}{event.rank}_'
                   f'score{("%.3f" % event.total_score).replace(".", "p")}.png')
    figure.clf()
    plt.close(figure)
    gc.collect()


# Makes the json file for a short event
def make_se_json(detector, event, times, energies, count_scints):
    event_path = f'{detector.get_export_loc()}/event_files/short_events/'
    tl.make_path(event_path)
    event_dict = dict()
    event_dict['SecondsOfDay'] = times[event.start:event.stop].tolist()
    event_dict['energies'] = energies[event.start:event.stop].tolist()
    event_dict['count_scintillator'] = count_scints[event.start:event.stop].tolist() if count_scints is not None else (
            [event.scintillator] * event.length)
    event_dict['lm_files'] = event.lm_files
    event_dict['trace_files'] = {scintillator: event.traces[scintillator].trace_name for scintillator in event.traces}

    # Saves the json file
    event_num_padding = '0' * (len(str(params.MAX_PLOTS_PER_SCINT)) - len(str(event.number)))
    rank_padding = '0' * (len(str(params.MAX_PLOTS_PER_SCINT)) - len(str(event.rank)))
    with open(
            f'{event_path}/'
            f'{dt.datetime.fromtimestamp(times[event.start] + detector.first_sec).strftime("%y%m%d_%H%M%S")}_'
            f'{event.scintillator}_'
            f'event{event_num_padding}{event.number}_'
            f'rank{rank_padding}{event.rank}_'
            f'score{("%.3f" % event.total_score).replace(".", "p")}.json', 'w') as file:
        json.dump(event_dict, file)


# Runs the short event search
def find_short_events(detector, modes, trace_dict, weather_cache, event_numbers=None):
    if modes['aircraft']:
        rollgap = params.AIRCRAFT_ROLLGAP
    elif modes['onescint'] or modes['allscints']:
        rollgap = params.INDIV_ROLLGAP
    else:
        # Calculating the rollgap adaptively based on the number of scintillators with data
        num_scintillators = 0
        for scintillator in detector:
            if detector.data_present_in(scintillator):
                num_scintillators += 1

        rollgap = round(params.INDIV_ROLLGAP + num_scintillators * params.ROLLGAP_CONTRIBUTION)

    for i in range(len(detector.scint_list)):
        if not detector.data_present_in(detector.scint_list[i]) or (not modes['allscints'] and i > 0):
            continue

        tl.print_logger('', detector.log)
        # Searching one scintillator at a time (onescint and allscints mode)
        if modes['onescint'] or modes['allscints']:
            scintillator = detector.scint_list[i]
            if not detector.data_present_in(scintillator):
                continue

            tl.print_logger(f'Searching eRC {detector.get_attribute(scintillator, "eRC")} '
                            f'({scintillator})...', detector.log)
            times = detector.get_lm_data(scintillator, 'SecondsOfDay')
            energies = detector.get_lm_data(scintillator, 'energies')
            count_scints = None

        # Normal operating mode (combining all scintillator data)
        else:
            scintillator = 'cs'  # Combined scintillators
            tl.print_logger('Searching combined scintillator data...', detector.log)
            (times,
             energies,
             count_scints) = tl.combine_data(detector)

        # Finding potential events with the search algorithm in find_short_events
        potential_events = short_event_search(detector, modes, scintillator, rollgap, times, energies)

        if len(potential_events) > 0:
            tl.print_logger('Generating scatter plots and event files...', detector.log)

            if event_numbers is not None and scintillator in event_numbers:
                plots_already_made = event_numbers[scintillator]
            else:
                plots_already_made = 0

            assert plots_already_made <= params.MAX_PLOTS_PER_SCINT
            if (plots_already_made + len(potential_events)) >= params.MAX_PLOTS_PER_SCINT:
                max_plots = params.MAX_PLOTS_PER_SCINT - plots_already_made
            else:
                max_plots = len(potential_events)

            # Scoring, numbering, and locating files for each event
            for j in range(len(potential_events)):
                event = potential_events[j]
                # Finds the file(s) that the event occurred in
                find_se_files(detector, event, times, count_scints)

                # Finds the trace(s) associated with the event and aligns them
                find_se_traces(detector, event, trace_dict, times, count_scints)

                # Calculates and records the event's subscores and final score
                calculate_se_score(detector, event, weather_cache, times, energies)

                event.number = plots_already_made + j + 1

            # Ranking each event
            ranked_events = sorted(potential_events,
                                   key=lambda x: -x.total_score)  # negative so we get a descending order sort
            for j in range(len(ranked_events)):
                ranked_events[j].rank = j + 1

            # Recording each event
            print('', file=detector.log)
            print('Potential short events:', file=detector.log)

            plots_made = 0
            if max_plots == 1:
                print(f'Making 1 plot and event file...')
            else:
                print(f'Making {max_plots} plots and event files...')

            for event in potential_events:
                # Stops making plots/event files/log entries after max reached
                # This is to prevent the program getting stuck on lab test days (with hundreds of thousands of "events")
                if plots_made == max_plots:
                    print(f'Max number of loggable events ({params.MAX_PLOTS_PER_SCINT}) reached.',
                          file=detector.log)
                    break

                # Logs the event
                start_second = times[event.start] - params.SEC_PER_DAY if (
                        times[event.start] > params.SEC_PER_DAY) else times[event.start]
                print(f'{dt.datetime.fromtimestamp(times[event.start] + detector.first_sec, dt.UTC)} UTC '
                      f'({start_second} seconds of day) - weather: {event.weather_conditions}',
                      file=detector.log)
                print(f'    Score: {event.total_score}, Rank: {event.rank}, Subscores: '
                      f'[Length: {event.len_subscore}, '
                      f'Clumpiness: {event.clumpiness_subscore}, '
                      f'HEL: {event.hel_subscore}, '
                      f'Weather: {event.weather_subscore}]\n',
                      file=detector.log)

                # Makes the scatter plot for the event
                make_se_scatterplot(detector, event, times, energies, count_scints)

                # Makes the json file for the event
                make_se_json(detector, event, times, energies, count_scints)

                plots_made += 1

            if event_numbers is not None:
                event_numbers[scintillator] = plots_made + plots_already_made

        tl.print_logger('', detector.log)

    gc.collect()


# Returns the list of preferred scintillators to be used in the long event search depending on the Detector
def get_le_scint_prefs(detector):
    try:
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/config/search_config.json', 'r') as file:
            all_preferences = json.load(file)['long_event_search_scints']
    except json.decoder.JSONDecodeError:
        raise SyntaxError('invalid syntax in search config file.')

    if detector.unit in all_preferences:
        preferences = []
        for scintillator in all_preferences[detector.unit]:
            if detector.is_valid_scintillator(scintillator):
                preferences.append(scintillator)

        return preferences

    return []


# Makes histogram used in long event search. If possible, cuts out counts below a certain energy
def make_le_hist(detector, scintillator, hist_end, bin_size):
    # Rounding the edge down to the nearest full bin
    hist_end -= hist_end % bin_size
    bins_allday = np.arange(0, hist_end + bin_size, bin_size)
    times = detector.get_lm_data(scintillator, 'SecondsOfDay')
    energies = detector.get_lm_data(scintillator, 'energies')
    if scintillator == 'NaI':
        times = np.delete(times, np.where(energies < params.NAI_CHANNEL_CUTOFF))
    else:
        times = np.delete(times, np.where(energies < params.LPL_CHANNEL_CUTOFF))

    hist_allday = np.histogram(times, bins=bins_allday)[0]
    return bins_allday[:-1], hist_allday  # Bins is always longer than hist by one for some reason


# Calculates mean for the long event search algorithm
def calculate_mue(hist_allday):
    hist_sum = 0
    hist_allday_nz = []
    for bin_val in hist_allday:
        if bin_val > 0:
            hist_allday_nz.append(bin_val)
            hist_sum += bin_val

    mue_val = hist_sum / len(hist_allday_nz)  # Mean number of counts/bin

    # Removes outliers (3 sigma) and recalculates mue
    abs_zscores = [np.abs((s - mue_val) / np.sqrt(mue_val)) for s in hist_allday_nz]
    sorting_order = np.argsort(abs_zscores)
    # Finds the index of the first zscore above 3 (this is essentially just a modified binary search algorithm)
    low = 0
    high = len(sorting_order) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if abs_zscores[sorting_order[mid]] >= 3:
            high = mid - 1
        else:
            low = mid + 1

    # Recalculates mue without outliers
    if low > 0:
        hist_sum = 0
        for i in range(0, low):
            hist_sum += hist_allday_nz[sorting_order[i]]

        return hist_sum / low
    else:  # No outliers
        return mue_val


# Calculates normal rolling baseline
def normal_baseline(mue, hist_allday, bin_size):
    window_size = params.N_WINDOW_SIZE // bin_size
    data_gap_locs = np.where(hist_allday == 0)[0]
    # Savgol can't handle data gaps, so here we find sections where data exists and only apply it to those
    # Checks against window size are necessary because savgol only accepts arrays of >= window size
    if len(data_gap_locs) > 0:
        prev = data_gap_locs[0]
        # Data present at the very beginning of the day
        if prev != 0 and prev >= window_size:
            new_mue = sp.signal.savgol_filter(hist_allday[0: prev], window_size, params.POLY_ORDER)
            for i in range(len(new_mue)):
                mue[i] = new_mue[i]

        # Data in the middle of the day
        for i in range(1, len(data_gap_locs)):
            curr = data_gap_locs[i]
            diff = curr - prev
            if diff > 1 and diff - 1 >= window_size:
                data_start = prev + 1
                new_mue = sp.signal.savgol_filter(hist_allday[data_start: curr],
                                                  window_size, params.POLY_ORDER)
                for j in range(len(new_mue)):
                    mue[j + data_start] = new_mue[j]

            prev = curr

        # Data present at the very end of the day
        data_start = prev + 1
        if data_start != len(hist_allday) and (len(hist_allday) - data_start) >= window_size:
            new_mue = sp.signal.savgol_filter(hist_allday[data_start:], window_size, params.POLY_ORDER)
            for i in range(len(new_mue)):
                mue[i + data_start] = new_mue[i]

    else:
        mue = sp.signal.savgol_filter(hist_allday, window_size, params.POLY_ORDER)

    return mue


# Calculates rolling baseline for aircraft mode
def aircraft_baseline(mue, bins_allday, hist_allday, bin_size):
    max_index = len(bins_allday) - 1
    # Getting the window size and gap in units of bins
    window_size = params.A_WINDOW_SIZE // bin_size
    window_gap = params.A_WINDOW_GAP // bin_size
    center_index = 0

    l_bins = []
    l_counts = []
    l_bool = []
    l_zeros = 0

    r_bins = []
    r_counts = []
    r_bool = []
    r_zeros = 0

    # Setting up the initial right window:
    # First bin (no neighbors, so it's a special case)
    r_index = window_gap + 1  # + center_index (zero at this stage, so we'll skip the extra operation)
    if hist_allday[window_gap + 1] > 0:
        r_bins.append(bins_allday[r_index])
        r_counts.append(hist_allday[r_index])
        r_bool.append(True)
    else:
        r_zeros += 1
        r_bool.append(False)

    r_index += 1
    # All the other bins
    # We've added one bin so far, and the very last one will be done by the beginning of the main loop below this
    # section, hence window_size - 2
    for i in range(window_size - 2):
        # Bin gets added to the window if both it AND its neighbors are nonzero
        if hist_allday[r_index] > 0 and hist_allday[r_index - 1] > 0 and hist_allday[r_index + 1] > 0:
            r_bins.append(bins_allday[r_index])
            r_counts.append(hist_allday[r_index])
            r_bool.append(True)
        else:
            r_zeros += 1
            r_bool.append(False)

        r_index += 1

    # Traversing the bins:
    l_index = center_index - (window_gap + 1)
    r_index = center_index + window_gap + window_size
    while center_index < len(bins_allday):
        # Advancing the left window:
        # Adding to the front of the window
        if l_index >= 0:
            # Bin gets added to the window if both it AND its neighbors are nonzero
            if (hist_allday[l_index] > 0 and
                    (l_index == 0 or (hist_allday[l_index - 1] > 0) and hist_allday[l_index + 1] > 0)):
                l_bins.append(bins_allday[l_index])
                l_counts.append(hist_allday[l_index])
                l_bool.append(True)
            else:
                l_bool.append(False)
                l_zeros += 1

        # Removing from the back of the window
        if len(l_bool) > window_size:
            if l_bool[0]:
                l_bins.pop(0)
                l_counts.pop(0)
            else:
                l_zeros -= 1

            l_bool.pop(0)

        # Advancing the right window:
        # Adding to the front of the window
        if r_index < len(bins_allday):
            if (hist_allday[r_index] > 0 and
                    (hist_allday[r_index - 1] > 0 and (r_index == max_index or hist_allday[r_index + 1] > 0))):
                r_bins.append(bins_allday[r_index])
                r_counts.append(hist_allday[r_index])
                r_bool.append(True)
            else:
                r_bool.append(False)
                r_zeros += 1

        # Removing from the back of the window
        if len(r_bool) > 0 and center_index > 0:
            if r_bool[0]:
                r_bins.pop(0)
                r_counts.pop(0)
            else:
                r_zeros -= 1

            r_bool.pop(0)

        # Only doing a fit for nonzero bins, and optimize curve fit needs at least three values
        if hist_allday[center_index] and len(l_bins) + len(r_bins) >= 3:
            # Fitting curve is linear if either of the windows contains a zero-value bin
            fit_curve = tl.o1_poly if l_zeros or r_zeros else tl.o2_poly
            with warnings.catch_warnings():
                # Context manager to catch and ignore the warning that's raised when curve_fit can't calculate
                # uncertainties on the fitting parameters for whatever reason
                warnings.simplefilter('ignore')
                fit = sp.optimize.curve_fit(fit_curve, l_bins + r_bins, l_counts + r_counts)[0]

            if fit_curve == tl.o2_poly:
                # Float casting is necessary for later parts of the day due to integer overflow
                mue[center_index] = fit_curve(float(bins_allday[center_index]), fit[0], fit[1], fit[2])
            else:
                mue[center_index] = fit_curve(bins_allday[center_index], fit[0], fit[1])

            # If the fit overestimates past the x-axis
            if mue[center_index] <= 0:
                mue[center_index] = mue[center_index - 1]

        center_index += 1
        l_index += 1
        r_index += 1

    return mue


# Long event search algorithm
def long_event_search(modes, day_bins, hist_allday, mue, sigma):
    z_scores = (hist_allday - mue) / sigma
    z_flags = np.where(z_scores > params.FLAG_THRESH)[0]  # Flags only those z-scores > params.FLAG_THRESH

    # Sorts z-flags into actual potential glows
    potential_glows = []
    glow_start = 0
    glow_length = 0
    previous_flag = -1
    for i in range(len(z_flags)):
        flag = z_flags[i]
        prev_bin = flag - 1
        if glow_length == 0:  # First zscore only
            glow_start = flag
            glow_length += 1
        elif prev_bin == previous_flag:  # Adding to the length of the glow
            glow_length += 1
        elif prev_bin > previous_flag:  # Recording the previous glow and starting to record a new one
            potential_glows.append(LongEvent(glow_start, glow_length, z_scores, day_bins))
            glow_start = flag
            glow_length = 1

        previous_flag = flag

    # Recording the last glow (if there is one)
    if glow_length > 0:
        potential_glows.append(LongEvent(glow_start, glow_length, z_scores, day_bins))

    # Rejects events whose bins don't have enough counts (aircraft only)
    if modes['aircraft']:
        a_potential_glows = []
        for glow in potential_glows:
            for i in range(glow.start, glow.stop):
                if hist_allday[i] >= params.LONG_EVENT_MIN_COUNTS:
                    a_potential_glows.append(glow)
                    break

        potential_glows = a_potential_glows

    return potential_glows


# Finds the list mode file(s) associated with a long event
def find_le_files(detector, le_scint_list, event):
    lm_files = {}
    for scintillator in le_scint_list:
        lm_files[scintillator] = detector.find_lm_file(scintillator, event.start_sec)

    return lm_files


# Makes the text file for a long event
def make_le_txt(detector, event, bin_size, event_num):
    event_path = f'{detector.get_export_loc()}/event_files/long_events/{bin_size}_sec_bins/'
    tl.make_path(event_path)
    timestamp = dt.datetime.fromtimestamp(event.start_sec + detector.first_sec, dt.UTC)
    with open(f'{event_path}/'
              f'{timestamp.strftime("%y%m%d_%H%M%S")}_'
              f'event{event_num}_'
              f'zscore{int(event.highest_score)}.txt', 'w') as event_file:

        print(f'{timestamp} UTC ({event.start_sec} '
              f'seconds of day), {event.stop_sec - event.start_sec} seconds long, highest z-score: '
              f'{event.highest_score}, {bin_size} sec bins',
              file=event_file)
        for scintillator in event.lm_files:
            print(f'{scintillator}: {event.lm_files[scintillator]}', file=event_file)


# Runs the long event search
def find_long_events(detector, modes, le_scint_list, bins_allday, hist_allday):
    bin_size = int(bins_allday[1] - bins_allday[0])  # Bin size in seconds
    # Note: mue is an array full of the mue values for each bin
    mue_val = calculate_mue(hist_allday)
    mue = np.full(len(bins_allday), mue_val)

    if modes['aircraft']:
        mue = aircraft_baseline(mue, bins_allday, hist_allday, bin_size)
    else:
        mue = normal_baseline(mue, hist_allday, bin_size)

    sigma = np.sqrt(mue)

    # Finding potential events with the search algorithm in find_long_events
    potential_glows = long_event_search(modes, bins_allday, hist_allday, mue, sigma)

    # Making histogram
    figure = plt.figure(figsize=[20, 11.0])
    figure.suptitle(f'{detector.unit} at {detector.deployment["location"]}, {detector.full_date_str}, '
                    f'{bin_size} sec bins')

    ax1 = figure.add_subplot(5, 1, 1)  # Daily histogram
    # The 4 top events in descending order (if they exist)
    ax2 = figure.add_subplot(5, 1, 2)
    ax3 = figure.add_subplot(5, 1, 3)
    ax4 = figure.add_subplot(5, 1, 4)
    ax5 = figure.add_subplot(5, 1, 5)

    ax1.bar(bins_allday, hist_allday, alpha=params.LE_MAIN_BAR_ALPHA,
            color=params.LE_MAIN_BAR_COLOR, width=bin_size)
    ax1.set_xlabel('Seconds of Day (UT)')
    ax1.set_ylabel('Counts/bin')
    ax1.plot(bins_allday, mue + params.FLAG_THRESH * sigma, color=params.LE_THRESH_LINE_COLOR, linestyle='dashed')

    # Creates legend
    allday_data = patches.Patch(color=params.LE_MAIN_BAR_COLOR, label='All Energies')
    allday_thresh_sigma = patches.Patch(color=params.LE_THRESH_LINE_COLOR,
                                        label=f'{params.FLAG_THRESH} Sigma Above All Energies', linestyle='dashed')
    ax1.legend(handles=[allday_data, allday_thresh_sigma, ], bbox_to_anchor=(1.05, 1), loc=1,
               borderaxespad=0.)
    ax1.grid(True)

    # Making event files and subplots (if they exist)
    if len(potential_glows) > 0:
        tl.print_logger('Generating event files...', detector.log)
        print('', file=detector.log)
        print('Potential glows:', file=detector.log)
        if len(potential_glows) == 1:
            print('Making 1 event file...')
        else:
            print(f'Making {len(potential_glows)} event files...')

        for i in range(len(potential_glows)):
            glow = potential_glows[i]

            # Getting the list mode files associated with the event
            glow.lm_files = find_le_files(detector, le_scint_list, glow)

            # Logging the event
            print(f'{dt.datetime.fromtimestamp(glow.start_sec + detector.first_sec, dt.UTC)} UTC ({glow.start_sec} '
                  f'seconds of day), {glow.stop_sec - glow.start_sec} seconds long, highest z-score: '
                  f'{glow.highest_score}, {bin_size} sec bins',
                  file=detector.log)

            # Making the event file
            make_le_txt(detector, glow, bin_size, i + 1)

        print('', file=detector.log)

        # Sorts the glows in descending order depending on their highest z-scores
        potential_glows = sorted(potential_glows, key=lambda x: -x.highest_score)  # Negative for descending order sort

        # Makes the histogram subplots
        ax_list = [ax2, ax3, ax4, ax5]
        for i in range(len(ax_list)):
            if i == len(potential_glows):
                break

            glow = potential_glows[i]
            ax = ax_list[i]
            left = 0 if (glow.peak_index - params.LE_SUBPLOT_PADDING) < 0 else \
                        glow.peak_index - params.LE_SUBPLOT_PADDING
            right = len(bins_allday) if glow.peak_index + params.LE_SUBPLOT_PADDING > len(bins_allday) else \
                glow.peak_index + params.LE_SUBPLOT_PADDING

            sub_bins = bins_allday[left:right]
            ax.bar(sub_bins, hist_allday[left:right], alpha=params.LE_SUBPLOT_BAR_ALPHA,
                   color=params.LE_SUBPLOT_BAR_COLOR, width=bin_size)
            ax.set_xlabel('Seconds of Day (UT)')
            ax.set_ylabel('Counts/bin')
            ax.plot(sub_bins, mue[left:right] + params.FLAG_THRESH * sigma[left:right],
                    color=params.LE_THRESH_LINE_COLOR, linestyle='dashed', linewidth=2)
            ax.grid(True)

    else:
        tl.print_logger(f'No glows found', detector.log)

    figure.tight_layout()

    # Saves the histogram(s):
    tl.print_logger('Saving Histogram...', detector.log)
    hist_path = f'{detector.get_export_loc()}'
    tl.make_path(hist_path)
    figure.savefig(f'{hist_path}/'
                   f'{detector.date_str}_'
                   f'histogram_{bin_size}_sec_bins_'
                   f'{len(potential_glows)}_events.png',
                   dpi=500)
    figure.clf()
    plt.close(figure)
    gc.collect()


# Mapping traces to their corresponding list mode files based on their timestamps
def map_traces(lm_filelist, trace_filelist):
    if len(trace_filelist) == 0 or len(lm_filelist) == 0:
        return {}

    lm_index = 0
    trace_index = 0
    trace_map = {}
    while lm_index < len(lm_filelist) - 1:
        next_file_timestamp = int(tl.file_timestamp(lm_filelist[lm_index + 1]))
        file_traces = []
        while trace_index < len(trace_filelist):
            trace_timestamp = int(tl.file_timestamp(trace_filelist[trace_index]))
            # If we start from the beginning of the trace filelist, every trace is either during or after the current
            # file.
            if trace_timestamp < next_file_timestamp:
                file_traces.append(trace_filelist[trace_index])
                trace_index += 1
                continue

            break

        if len(file_traces) > 0:
            trace_map[lm_filelist[lm_index]] = file_traces

        lm_index += 1

    # The remaining traces (if there are any) must be during the last file
    if trace_index != len(trace_filelist):
        trace_map[lm_filelist[-1]] = trace_filelist[trace_index:len(trace_filelist)]

    return trace_map


# Returns a list of indices for each partition point defining the set of files during or after the point
def get_lm_sets(partition_points, lm_filelist):
    set_indices = [0] * len(partition_points)
    set_indices[-1] = len(lm_filelist)  # The last partition point must contain no files
    part_index = 1  # The first partition point must contain all files, so its index stays zero
    lm_index = 0
    while part_index < len(partition_points) - 1:
        if lm_index < len(lm_filelist):
            while lm_index < len(lm_filelist):
                if int(tl.file_timestamp(lm_filelist[lm_index])) >= partition_points[part_index]:
                    break

                lm_index += 1

            set_indices[part_index] = lm_index
        else:
            set_indices[part_index] = lm_index

        part_index += 1

    return set_indices


# Returns the total memory in bytes of all files contained in the partition defined by points start and end
def get_partition_memory(detector, scintillator, start, end, trace_map, lm_filelist):
    memory = 0
    for i in range(start, end):
        file = lm_filelist[i]
        memory += tl.file_size(file) * detector.lm_growth_factors[scintillator]
        if file in trace_map:
            for trace in trace_map[file]:
                memory += tl.file_size(trace) * detector.trace_growth_factors[scintillator]

    return memory


# Returns lists of all files contained in the partition defined by points start and end
def get_partition_files(start, end, trace_map, lm_filelist):
    lm_partition = lm_filelist[start:end]
    trace_partition = []
    for file in lm_partition:
        if file in trace_map:
            trace_partition += trace_map[file]

    return lm_partition, trace_partition


# Partitions the day into self-contained detector objects depending on the memory limit in parameters
def make_chunks(detector):
    allowed_memory = psutil.virtual_memory()[1] * get_max_mem_frac()

    lm_filelists = {scintillator: detector.get_attribute(scintillator, 'lm_filelist')
                    for scintillator in detector}

    # Making the partition points using the filelist of the highest priority scintillator. Using the times of files
    # ensures that we're partitioning the day according to the data's approximate density
    best_scintillator = ''
    for scintillator in detector:
        if len(lm_filelists[scintillator]) > 0:
            best_scintillator = scintillator
            break

    if best_scintillator == '':
        raise FileNotFoundError('no data present.')

    partition_points = ([0] + [int(tl.file_timestamp(file)) for file in lm_filelists[best_scintillator]] +
                        [240000])

    # For each scintillator, mapping traces to their corresponding list mode files and mapping sets of the list
    # mode files to all the partition points
    trace_maps = {}
    all_lm_sets = {}
    for scintillator in detector:
        lm_filelist = lm_filelists[scintillator]
        trace_filelist = detector.get_attribute(scintillator, 'trace_filelist')
        trace_maps[scintillator] = map_traces(lm_filelist, trace_filelist)
        all_lm_sets[scintillator] = get_lm_sets(partition_points, lm_filelist)

    # Partitioning the day into chunks based on the amount of memory each partition point adds
    chunk_list = []
    total_memory = 0
    prev_point = 0
    for i in range(1, len(partition_points) - 1):
        new_memory = 0
        for scintillator in detector:
            trace_map = trace_maps[scintillator]
            lm_sets = all_lm_sets[scintillator]
            lm_filelist = lm_filelists[scintillator]
            new_memory += get_partition_memory(detector, scintillator, lm_sets[i], lm_sets[i + 1],
                                               trace_map, lm_filelist)

        total_memory += new_memory
        # Making a new chunk if the amount of new memory added by partitioning at i + 1 takes us over the limit
        if total_memory > allowed_memory:
            chunk = detector.get_clone()
            for scintillator in detector:
                trace_map = trace_maps[scintillator]
                lm_sets = all_lm_sets[scintillator]
                lm_filelist = lm_filelists[scintillator]
                lm_partition, trace_partition = get_partition_files(lm_sets[prev_point], lm_sets[i],
                                                                    trace_map, lm_filelist)
                chunk.set_attribute(scintillator, 'lm_filelist', lm_partition, deepcopy=False)
                chunk.set_attribute(scintillator, 'trace_filelist', trace_partition, deepcopy=False)

            chunk_list.append(chunk)
            total_memory = new_memory
            prev_point = i

    # Setting up the last chunk with whatever is left
    chunk = detector.get_clone()
    for scintillator in detector:
        trace_map = trace_maps[scintillator]
        lm_sets = all_lm_sets[scintillator]
        lm_filelist = lm_filelists[scintillator]
        lm_partition, trace_partition = get_partition_files(lm_sets[prev_point], lm_sets[-1],
                                                            trace_map, lm_filelist)
        chunk.set_attribute(scintillator, 'lm_filelist', lm_partition, deepcopy=False)
        chunk.set_attribute(scintillator, 'trace_filelist', trace_partition, deepcopy=False)

    chunk_list.append(chunk)
    return chunk_list


# Gets necessary info from command line args and then runs the program
def main():
    if len(sys.argv) >= 4:
        first_date = str(sys.argv[1])
        second_date = str(sys.argv[2])
        unit = str(sys.argv[3])
    else:
        print('Error: please provide a first date, a second date, and a unit name.')
        exit()

    # Makes sure that inputs are valid
    if not is_valid_search(first_date, second_date, unit, print_feedback=True):
        exit()

    if len(sys.argv) > 4:
        mode_info = sys.argv[4:]
    else:
        mode_info = []

    program(first_date, second_date, unit, mode_info)


# Main program function
def program(first_date, second_date, unit, mode_info):
    matplotlib.use('Agg')  # Memory leaks without this
    modes = get_modes(mode_info)

    # Looping through the dates
    for date_str in tl.make_date_list(first_date, second_date):
        low_memory_mode = False
        print('')
        print(f'{tl.short_to_full_date(date_str)}:')

        # Initializes the detector object
        print('Importing data...')
        detector = get_detector(unit, date_str)
        try:
            # Tells the detector to use custom import/export directories if the user asks for it
            if modes['custom']:
                index = mode_info.index('-c')
                if index + 2 < len(mode_info):
                    import_index = index + 1
                    if mode_info[import_index] != 'none' and mode_info[import_index] != '/':
                        detector.set_import_loc(mode_info[import_index])

                    export_index = index + 2
                    if mode_info[export_index] != 'none' and mode_info[export_index] != '/':
                        detector.set_export_loc(mode_info[export_index])

            if not detector.has_identity():
                raise FileNotFoundError("couldn't infer identity.")

        except FileNotFoundError:
            print('Error: no data files to infer detector identity from. Please provide the import location of '
                  'the data.')
            exit()

        # Logs relevant data files and events in a .txt File
        log_path = f'{detector.get_export_loc()}'
        tl.make_path(log_path)
        log = open(f'{log_path}/log.txt', 'w')
        print(f'{tl.short_to_full_date(date_str)}:', file=log)

        # Normal operating mode
        try:
            # Imports the data
            if modes['pickle']:  # In pickle mode: reads the pickle file, or exports one if it doesn't exist yet
                pickle_paths = glob.glob(f'{detector.get_export_loc()}/detector.pickle')
                if len(pickle_paths) > 0:
                    detector = tl.unpickle_detector(pickle_paths[0])
                    detector.log = log
                else:
                    detector.log = log
                    detector.import_data(clean_energy=modes['clnenrg'], feedback=True,
                                         mem_frac=get_max_mem_frac())
                    tl.pickle_detector(detector, 'detector')
            else:
                detector.log = log
                detector.import_data(clean_energy=modes['clnenrg'], feedback=True,
                                     mem_frac=get_max_mem_frac())

            print('')
            print('Done.')

            # Trace Search
            if not modes['sktrace']:
                tl.print_logger('\n', detector.log)
                tl.print_logger('Filtering traces...', detector.log)
                trace_dict = find_traces(detector)
                tl.print_logger('Done.', detector.log)
            else:
                trace_dict = {scintillator: [] for scintillator in detector}

            # Checks to see that necessary list mode data is present
            if not detector:
                raise FileNotFoundError('no data present.')

            # Short event search
            if not modes['skshort']:
                tl.print_logger('\n', detector.log)
                tl.print_logger('Starting search for short events...', detector.log)

                # Calling the short event search algorithm
                weather_cache = {}
                find_short_events(detector, modes, trace_dict, weather_cache)

                tl.print_logger('Done.', detector.log)

            # Long event search
            if not modes['skglow']:
                tl.print_logger('\n', detector.log)
                tl.print_logger('Starting search for glows...', detector.log)

                # Choosing whether to use preferred scintillators for the long event search based on whether they have
                # data present
                le_scint_list = []
                for scintillator in get_le_scint_prefs(detector):
                    if detector.data_present_in(scintillator):
                        le_scint_list.append(scintillator)

                # In the case that either 1) there aren't any preferred scintillators or 2) none of the preferred
                # scintillators have data present, we instead use the highest priority scintillator with data present
                if len(le_scint_list) == 0:
                    for scintillator in detector:
                        if detector.data_present_in(scintillator):
                            le_scint_list.append(scintillator)
                            break

                print(f'Using the following scintillators: {", ".join(le_scint_list)}', file=detector.log)

                # Determining the end point of the histogram based on the last counts of each scintillator. Using
                # the earliest to ensure that the last bin has data from all scintillators
                hist_end = float('inf')
                for scintillator in le_scint_list:
                    last_count = detector.get_attribute(scintillator, 'lm_file_ranges', deepcopy=False)[-1][1]
                    if last_count < hist_end:
                        hist_end = last_count

                hist_end = int(hist_end)

                for bin_size in [params.SHORT_BIN_SIZE, params.LONG_BIN_SIZE]:
                    tl.print_logger('', detector.log)
                    tl.print_logger(f'Searching with {bin_size} second bins...', detector.log)
                    # Makes daily histogram with each scintillator's contribution
                    bins_allday = None
                    hist_allday = None
                    for scintillator in le_scint_list:
                        bins_allday, scint_hist = make_le_hist(detector, scintillator, hist_end, bin_size)
                        hist_allday = scint_hist if hist_allday is None else hist_allday + scint_hist

                    # Calling the long event search algorithm
                    find_long_events(detector, modes, le_scint_list, bins_allday, hist_allday)

                tl.print_logger('', detector.log)
                tl.print_logger('Done.', detector.log)

        except MemoryError:
            low_memory_mode = True

        except FileNotFoundError:  # Missing necessary data
            tl.print_logger('\n', detector.log)
            tl.print_logger('No/missing data for specified day.', detector.log)

        except Exception as ex:  # Logging errors
            log_error(detector, modes, ex)

        # Low memory mode
        if low_memory_mode:
            chunk_path_list = []
            try:
                tl.print_logger('', detector.log)
                tl.print_logger('Not enough memory. Entering low memory mode...', detector.log)

                # Clears leftover data (just to be sure)
                detector.clear(clear_filelists=False)

                # Makes and sets up the chunks
                chunk_list = make_chunks(detector)
                num_chunks = len(chunk_list)
                for chunk in chunk_list:
                    chunk.log = log

                chunk_scint_list = chunk_list[0].scint_list

                # Imports data to each chunk and then pickles the chunks (and checks that data is actually present)
                print('Importing data...')

                chunk_num = 1
                scint_data = {scintillator: False for scintillator in detector}
                # Passes reader objects between chunks
                reader_dict = {scintillator: chunk_list[0].get_attribute(scintillator, 'reader', deepcopy=False)
                               for scintillator in chunk_scint_list}

                while chunk_list:
                    chunk = chunk_list.pop(0)
                    # Updates chunk to include previous chunk's reader
                    if chunk_num != 1:
                        for scintillator in chunk:
                            chunk.set_attribute(scintillator, 'reader', reader_dict[scintillator], deepcopy=False)

                    tl.print_logger('', detector.log)
                    tl.print_logger(f'Chunk {chunk_num} (of {num_chunks}):', detector.log)
                    chunk.import_data(existing_filelists=True, clean_energy=modes['clnenrg'], feedback=True)

                    # Keeping track of which scintillators have data in at least one chunk
                    for scintillator in chunk:
                        if not scint_data[scintillator]:
                            scint_data[scintillator] = chunk.data_present_in(scintillator)

                    # Makes a full list of filetime extrema for long event search
                    for scintillator in chunk:
                        extrema = detector.get_attribute(scintillator, 'lm_file_ranges', deepcopy=False)
                        extrema += chunk.get_attribute(scintillator, 'lm_file_ranges')

                    # Pickle chunk and add its path to the list
                    chunk_path_list.append(tl.pickle_detector(chunk, f'chunk{chunk_num}'))

                    # Eliminates the chunk from active memory
                    del chunk

                    chunk_num += 1

                print('')
                print('Done.')

                # Trace Search
                chunk_trace_dicts = {}
                if not modes['sktrace']:
                    tl.print_logger('\n', detector.log)
                    tl.print_logger('Filtering traces...', detector.log)
                    for chunk_path in chunk_path_list:
                        chunk = tl.unpickle_detector(chunk_path)
                        trace_dict = find_traces(chunk)
                        chunk_trace_dicts[chunk_path] = trace_dict
                        del chunk

                    tl.print_logger('Done.', detector.log)
                else:
                    for chunk_path in chunk_path_list:
                        chunk_trace_dicts[chunk_path] = {scintillator: [] for scintillator in chunk_scint_list}

                # Skips the day if data is missing
                data_present = False
                for scintillator in scint_data:
                    if not data_present:
                        data_present = scint_data[scintillator]

                if not data_present:
                    raise FileNotFoundError('no data present.')

                # Short event search
                if not modes['skshort']:
                    tl.print_logger('\n', detector.log)
                    tl.print_logger('Starting search for short events...', detector.log)
                    tl.print_logger('Warning: in low memory mode, short events will be ranked on a per-chunk basis.',
                                    detector.log)
                    tl.print_logger('', detector.log)
                    event_numbers = {}
                    weather_cache = {}
                    chunk_num = 1
                    for chunk_path in chunk_path_list:
                        chunk = tl.unpickle_detector(chunk_path)
                        chunk.log = log
                        tl.print_logger(f'Chunk {chunk_num} (of {num_chunks}):', detector.log)
                        # Calling the short event search algorithm
                        find_short_events(chunk, modes, chunk_trace_dicts[chunk_path], weather_cache,
                                          event_numbers=event_numbers)
                        del chunk
                        chunk_num += 1

                    tl.print_logger('Done.', detector.log)

                # Long event search
                if not modes['skglow']:
                    tl.print_logger('\n', detector.log)
                    tl.print_logger('Starting search for glows...', detector.log)
                    le_scint_list = []
                    # Choosing whether to use preferred scintillators for the long event search based on whether they
                    # have data present in at least one chunk
                    for scintillator in get_le_scint_prefs(detector):
                        if scint_data[scintillator]:
                            le_scint_list.append(scintillator)

                    # In the case that either 1) there aren't any preferred scintillators or 2) none of the preferred
                    # scintillators have data present, we instead use the highest priority scintillator with
                    # data present
                    if len(le_scint_list) == 0:
                        for scintillator in scint_data:
                            if scint_data[scintillator]:
                                le_scint_list.append(scintillator)
                                break

                    # Determining the end point of the histogram based on the last counts of each scintillator. Using
                    # the earliest to ensure that the last bin has data from all scintillators
                    hist_end = float('inf')
                    for scintillator in le_scint_list:
                        last_count = detector.get_attribute(scintillator, 'lm_file_ranges', deepcopy=False)[-1][1]
                        if last_count < hist_end:
                            hist_end = last_count

                    hist_end = int(hist_end)

                    print(f'Using the following scintillators: {", ".join(le_scint_list)}', file=detector.log)
                    for bin_size in [params.SHORT_BIN_SIZE, params.LONG_BIN_SIZE]:
                        tl.print_logger('', detector.log)
                        tl.print_logger(f'Searching with {bin_size} second bins...', detector.log)
                        bins_allday = None
                        hist_allday = None
                        for chunk_path in chunk_path_list:
                            chunk = tl.unpickle_detector(chunk_path)
                            for scintillator in le_scint_list:
                                if chunk.data_present_in(scintillator):
                                    # Histograms the counts from each scintillator and combines them with the main one
                                    bins_allday, scint_hist = make_le_hist(chunk, scintillator, hist_end, bin_size)
                                    hist_allday = scint_hist if hist_allday is None else hist_allday + scint_hist

                            del chunk

                        # Calling the long event search algorithm
                        find_long_events(detector, modes, le_scint_list, bins_allday, hist_allday)

                    tl.print_logger('', detector.log)
                    tl.print_logger('Done.', detector.log)

            except MemoryError:
                tl.print_logger('\n', log)
                tl.print_logger('Error: cannot complete search. Too little memory available on system.', log)

            except FileNotFoundError:  # Missing necessary data
                tl.print_logger('\n', detector.log)
                tl.print_logger('No/missing data for specified day.', detector.log)

            except Exception as ex:  # Logging errors
                log_error(detector, modes, ex)

            finally:
                # Deletes chunk .pickle files
                for chunk_path in chunk_path_list:
                    os.remove(chunk_path)

        del detector
        log.close()


if __name__ == '__main__':
    main()
