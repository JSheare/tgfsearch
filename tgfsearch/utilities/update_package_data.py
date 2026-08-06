"""A module containing a utility that updates the tgfsearch package's built-in data."""
import os
from typing import Any, Dict

import tgfsearch.helpers.api as api
from tgfsearch.helpers.helper_funcs import write_json_file


def build_detector_config(piecewise_timeout: float = 60.0) -> Dict[str, Any] | None:
    """
    Returns an up-to-date detector config built with information from the API.

    Parameters
    ----------
    piecewise_timeout : float
        Optional. The maximum amount of time to wait for a response from the API each time that it is called.

    Returns
    -------
    Dict[str, Any] | None
        An up-to-date detector config, if all information was successfully retrieved. A None otherwise. See config
        file docs for details on its structure.

    """

    config = dict()

    # Storing the default data root
    default_data_root = api.get_data_root(timeout=piecewise_timeout)
    if default_data_root is None:
        return None

    config['default_data_root'] = default_data_root

    # Building the scintillator priority list
    scints = api.get_scintillators(timeout=piecewise_timeout)
    if scints is None:
        return None

    config['scintillator_priority'] = [scint for scint in scints]
    config['scintillator_priority'].sort(key=lambda scint: scints[scint]['scint_priority'])

    # Retrieving and storing instrument identities
    config['identities'] = dict()
    instruments = api.get_instruments(piecewise_timeout)
    if instruments is None:
        return None

    for instrument in api.get_instruments(piecewise_timeout):
        config['identities'][instrument] = dict()
        subdir = api.get_instrument_subdir(instrument, timeout=piecewise_timeout)
        if subdir is None:
            return None

        config['identities'][instrument]['subdir'] = subdir
        instrument_configs = api.get_instrument_config(instrument, '*', timeout=piecewise_timeout)
        if instrument_configs is None:
            return None

        for date in instrument_configs:
            for scint in instrument_configs[date]:
                instrument_configs[date][scint].pop('long_event_search')

        config['identities'][instrument]['scintillators'] = instrument_configs

    return config


def build_search_config(piecewise_timeout: float = 60.0) -> Dict[str, Any] | None:
    """
    Returns an up-to-date search config built with information from the API.

    Parameters
    ----------
    piecewise_timeout : float
        Optional. The maximum amount of time to wait for a response from the API each time that it is called.

    Returns
    -------
    Dict[str, Any] | None
        An up-to-date search config, if all information was successfully retrieved. A None otherwise. See config file
        docs for details on its structure.

    """

    config = dict()

    # Building the short event search plot color record
    config['short_event_search_colors'] = dict()
    scintillators = api.get_scintillators(timeout=piecewise_timeout)
    if scintillators is None:
        return None

    for scint in scintillators:
        config['short_event_search_colors'][scint] = scintillators[scint]['plot_color']

    # Building the long event search scintillator record
    config['long_event_search_scints'] = dict()
    instruments = api.get_instruments(timeout=piecewise_timeout)
    if instruments is None:
        return None

    for instrument in instruments:
        instrument_configs = api.get_instrument_config(instrument, '*', timeout=piecewise_timeout)
        if instrument_configs is None:
            return None

        long_event_search_scints = set()
        for date in instrument_configs:
            for scint in instrument_configs[date]:
                if instrument_configs[date][scint]['long_event_search']:
                    long_event_search_scints.add(scint)

        config['long_event_search_scints'][instrument] = list(long_event_search_scints)

    return config


def main() -> None:
    # Updating the config files
    config_loc = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config'
    if os.access(config_loc, os.W_OK):
        detector_config = build_detector_config()
        if detector_config is not None:
            write_json_file(detector_config, f'{config_loc}/detector_config.json', pretty=True)
        else:
            print('Failed to update detector config. No/improper response from API.')

        search_config = build_search_config()
        if search_config is not None:
            write_json_file(search_config, f'{config_loc}/search_config.json', pretty=True)
        else:
            print('Failed to update search config. No/improper response from API.')

    else:
        print('Failed to update config files. Insufficient permissions.')

    # Updating the deployment files
    deployments_loc = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/deployments/'
    if os.access(deployments_loc, os.W_OK):
        instruments = api.get_instruments()
        if instruments is None:
            print('Failed to update deployment files. No/improper response from API.')
        else:
            for instrument in instruments:
                deployments = api.get_instrument_deployment(instrument, '*')
                if deployments is not None:
                    for deployment in deployments:
                        write_json_file(deployment, f'{deployments_loc}/{instrument}_deployment_'
                                                    f'{deployment["start_date"]}_{deployment["end_date"]}.json')

    else:
        print('Failed to update deployment files. Insufficient permissions.')


if __name__ == '__main__':
    main()
