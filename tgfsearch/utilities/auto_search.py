"""A script that automatically runs searches on unexamined data."""
import glob as glob
import json as json
import os as os
import psutil as psutil
import sys as sys
import traceback as traceback

# Adds grandparent directory to sys.path. Necessary to make the imports below work when running this file as a script
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import tgfsearch.tools as tl
from tgfsearch.search import program


def main():
    """Optional argument order: results_directory, autosearch_misc_directory"""
    if len(sys.argv) >= 2:
        results_loc = sys.argv[1]
    else:
        results_loc = os.getcwd()

    if len(sys.argv) >= 3:
        auto_search_loc = sys.argv[2].replace('\\', '/')
    else:
        auto_search_loc = (os.getcwd() + '\\Autosearch').replace('\\', '/')

    tl.make_path(auto_search_loc)
    # Checks to see if the program is already running (maybe the dataset it's checking is quite large or long)
    try:
        with open(f'{auto_search_loc}/pid.txt', 'r') as existing_pid_file:
            if psutil.pid_exists(int(existing_pid_file.readline())):
                exit()

    except FileNotFoundError:
        pass

    # Makes the pid file
    with open(f'{auto_search_loc}/pid.txt', 'w') as pid_file:
        pid_file.write(str(os.getpid()))

    # Attempts to read the checked dates file
    try:
        with open(f'{auto_search_loc}/checked_dates.json', 'r') as date_file:
            checked_dates = json.load(date_file)

    except FileNotFoundError:
        checked_dates = {}

    # Reads the config file
    try:
        with open(f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config/auto_search_config.json',
                'r') as file:
            config = json.load(file)

    except json.decoder.JSONDecodeError:
        print('Error: Invalid syntax in auto search config file.')
        exit()

    for unit in config:
        for sub_entry in config[unit]:
            tree = config[unit][sub_entry]['tree']
            # The convention that I've chosen is that after date is inclusive and before date is not
            before_date = int(config[unit][sub_entry]['before_date'])
            after_date = int(config[unit][sub_entry]['after_date'])
            if unit not in checked_dates:
                checked_dates[unit] = []

            queue = []
            date_paths = glob.glob(f'{tree}/*')
            date_paths.sort()
            for date_path in date_paths:
                if len(date_path) >= 6:
                    date = date_path[-6:]
                    if (date.isnumeric() and len(os.listdir(date_path)) > 0 and (after_date <= int(date) < before_date)
                            and date not in checked_dates[unit]):
                        queue.append(date)

            # Checking the most recent stuff first
            queue.reverse()

            for date in queue:
                mode_info = list(config[unit][sub_entry]['mode_info'])  # Implicit copy
                # Adding custom import/export directories
                mode_info.append('-c')
                mode_info.append(f'{tree}/{date}')
                mode_info.append(results_loc)

                print(f'Running search: {date} {unit} {mode_info}...')
                try:
                    program(date, date, unit, mode_info)

                    # Adding the day to the list of checked dates
                    checked_dates[unit].append(date)

                    # Dumping the updated list of checked dates to a json file for use the next time the program runs
                    with open(f'{auto_search_loc}/checked_dates.json', 'w') as file:
                        json.dump(checked_dates, file)

                except Exception as ex:
                    # Writing any critical errors raised by the search to text files so that they can be
                    # examined/fixed later
                    print(f'Search encountered the following error: {ex}')
                    tl.make_path(f'{auto_search_loc}/error_logs')
                    with open(f'{auto_search_loc}/error_logs/{unit}_{date}_error.txt', 'w') as file:
                        file.write(traceback.format_exc())

    # Deletes the pid file
    os.remove(f'{auto_search_loc}/pid.txt')


if __name__ == '__main__':
    main()
