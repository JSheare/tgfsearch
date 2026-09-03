"""A module containing functions used to access the UCSC TGF group API."""
import json
import pandas
import requests
import time
from typing import Dict, Generator, List

import tgfsearch.config.parameters as params


def _exponential_backoff() -> Generator[float, None, None]:
    """Helper function implementing a generator that yields an exponentially increasing retry wait time up to a certain
    maximum number of retries."""
    attempt = 1
    while attempt <= params.API_MAX_RETRIES:
        yield params.API_BACKOFF_BASE_DELAY * 2**attempt
        attempt += 1

    yield 0.0


def get_data_root(timeout: float = 60.0) -> str | None:
    """Returns the root location for all TGF data on the main data computer.

    Parameters
    ----------
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    str | None
        The data root, if the API request was successful. A None otherwise.

    """

    for wait in _exponential_backoff():
        try:
            response = requests.get(f'{params.API_URL}/data-root/', timeout=timeout)
            response.raise_for_status()
            return response.json()['data_root']
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code == 429:
                if wait == 0:
                    break

                time.sleep(wait)
            else:
                return None

        except (requests.exceptions.RequestException, json.decoder.JSONDecodeError, KeyError):
            return None

    return None


def get_instruments(timeout: float = 60.0) -> List[str] | None:
    """Returns a list of all registered UCSC TGF group instruments.

    Parameters
    ----------
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    List[str] | None
        A list containing the names of all registered UCSC TGF group instruments, if the API request was successful. A
        None otherwise

    """

    for wait in _exponential_backoff():
        try:
            response = requests.get(f'{params.API_URL}/instruments/', timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code == 429:
                if wait == 0:
                    break

                time.sleep(wait)
            else:
                return None

        except (requests.exceptions.RequestException, json.decoder.JSONDecodeError, KeyError):
            return None

    return None


def get_scintillators(timeout: float = 60.0) -> Dict[str, Dict[str, str | int]] | None:
    """Returns a record of all registered scintillator types and their associated information.

    Parameters
    ----------
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    Dict[str, Dict[str, str | int]] | None
        A dictionary containing records for all registered scintillator types, if the API request was successful. A
        None otherwise. Each entry in the dictionary will be of the following form: {scint_name: {'scint_priority': x,
        'plot_color': x}}.

    Raises
    ------
    requests.exceptions.HTTPError
        If the API replied with an HTTP error.

    """

    for wait in _exponential_backoff():
        try:
            response = requests.get(f'{params.API_URL}/scintillators/', timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code == 429:
                if wait == 0:
                    break

                time.sleep(wait)
            else:
                return None

        except (requests.exceptions.RequestException, json.decoder.JSONDecodeError, KeyError):
            return None

    return None


def get_instrument_subdir(instrument: str, timeout: float = 60.0) -> str | None:
    """Returns the data subdirectory for a particular instrument (within the data root) on the UCSC TGF group's main
    data computer.

    Parameters
    ----------
    instrument : str
        The name of the instrument to get the subdirectory for.
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    str | None
        The instrument's data subdirectory, if the API request was successful. A None otherwise.

    Raises
    ------
    ValueError
        If the given instrument name is invalid.

    """

    for wait in _exponential_backoff():
        try:
            response = requests.get(f'{params.API_URL}/instrument-subdir/', params={'instrument': instrument},
                                    timeout=timeout)
            response.raise_for_status()
            return response.json()['subdir']
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code == 422:
                try:
                    feedback = ex.response.json()['detail']
                except (json.decoder.JSONDecodeError, KeyError):
                    feedback = 'one or more inputs is invalid.'

                raise ValueError(feedback)
            elif ex.response.status_code == 429:
                if wait == 0:
                    break

                time.sleep(wait)
            else:
                return None

        except (requests.exceptions.RequestException, json.decoder.JSONDecodeError, KeyError):
            return None

    return None


def get_instrument_config(instrument: str, date: str,
                          timeout: float = 60.0) -> Dict[str, Dict[str, Dict[str, str | bool]]] | None:
    """Returns the configuration information for the given instrument.

    Parameters
    ----------
    instrument : str
        The name of the instrument to get configuration information for.
    date : str
        The date to get configuration information for. This should be of the form 'YYMMDD', or an asterisk to get all
        configurations for the instrument.
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, str | bool]]] | None
        A dictionary containing the given instrument's configuration on the given date, if the API request was
        successful. A None otherwise. The dictionary contains instrument configurations after particular dates, and each
        entry has the following form: {after_date_1: {scint_name_1: {'erc': x, 'format_name': x,
        'long_event_search': x}, scint_name_2: ...}}.

    Raises
    ------
    ValueError
        If either the given instrument name or date are invalid.

    """

    for wait in _exponential_backoff():
        try:
            response = requests.get(f'{params.API_URL}/instrument-config/',
                                    params={'instrument': instrument, 'date': date}, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code == 422:
                try:
                    feedback = ex.response.json()['detail']
                except (json.decoder.JSONDecodeError, KeyError):
                    feedback = 'one or more inputs is invalid.'

                raise ValueError(feedback)
            elif ex.response.status_code == 429:
                if wait == 0:
                    break

                time.sleep(wait)
            else:
                return None

        except (requests.exceptions.RequestException, json.decoder.JSONDecodeError, KeyError):
            return None

    return None


def get_instrument_deployment(instrument: str, date: str, timeout: float = 60.0) -> List[Dict[str, str | float]] | None:
    """Returns deployment information for a particular instrument.

    Parameters
    ----------
    instrument : str
        The name of the instrument to get deployment information for.
    date : str
        The date to get deployment information for. This should be of the form 'YYMMDD', or an asterisk to get all
        deployments for the instrument.
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    List[Dict[str, str | float]] | None
        A list of dictionaries containing deployments that match the given instrument and date, if the API request was
        successful. A None otherwise. Each deployment dictionary is of the following form: {'instrument': instrument,
        'start_date': YYMMDD, 'end_date': YYMMDD, 'location': x, 'tz_identifier': x, 'weather_station': x,
        'sounding_station': x, 'latitude': x, 'longitude': x, 'altitude': x, 'notes': x}.

    Raises
    ------
    ValueError
        If either the given instrument name or date are invalid.

    """

    for wait in _exponential_backoff():
        try:
            response = requests.get(f'{params.API_URL}/instrument-deployment/',
                                    params={'instrument': instrument, 'date': date}, timeout=timeout)
            response.raise_for_status()
            deployments = response.json()
            for i in range(0, len(deployments)):
                deployments[i]['instrument'] = instrument

            return deployments
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code == 422:
                try:
                    feedback = ex.response.json()['detail']
                except (json.decoder.JSONDecodeError, KeyError):
                    feedback = 'one or more inputs is invalid.'

                raise ValueError(feedback)
            elif ex.response.status_code == 429:
                if wait == 0:
                    break

                time.sleep(wait)
            else:
                return None

        except (requests.exceptions.RequestException, json.decoder.JSONDecodeError, KeyError):
            return None

    return None


def get_weather(instrument: str, date: str, timeout: float = 60.0) -> List[Dict[str, float | str]] | None:
    """Returns weather information for the given instrument on the given date.

    Parameters
    ----------
    instrument : str
        The name of the instrument to get weather information for.
    date : str
        The date to get weather information for. This should be of the form 'YYMMDD'.
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    List[Dict[str, float | str]] | None
        A list of weather measurements for the given instrument on the given date, if the API request was successful.
        A None otherwise. Each measurement has the following form {'measurement_time': x_epoch, 'condition': x}.

    Raises
    ------
    ValueError
        If either the given instrument name or date are invalid.

    """

    for wait in _exponential_backoff():
        try:
            response = requests.get(f'{params.API_URL}/weather/', params={'instrument': instrument, 'date': date},
                                    timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code == 422:
                try:
                    feedback = ex.response.json()['detail']
                except (json.decoder.JSONDecodeError, KeyError):
                    feedback = 'one or more inputs is invalid.'

                raise ValueError(feedback)
            elif ex.response.status_code == 429:
                if wait == 0:
                    break

                time.sleep(wait)
            else:
                return None

        except (requests.exceptions.RequestException, json.decoder.JSONDecodeError, KeyError):
            return None

    return None


def get_weather_table(instrument: str, date: str, timeout: float = 60.0) -> pandas.DataFrame | None:
    """Returns a weather table for the given instrument on the given date.

    Parameters
    ----------
    instrument : str
        The name of the instrument to get weather information for.
    date : str
        The date to get weather information for. This should be of the form 'YYMMDD'.
    timeout : float
        Optional. The maximum amount of time to wait for a response from the API.

    Returns
    -------
    pandas.core.frame.DataFrame | None
        A pandas dataframe containing weather information for the passed instrument on the passed date, if the API
        request was successful. A None otherwise. The dataframe has two columns: 'measurement_time', which contains
        the time of the measurement in epoch, and 'condition', which contains the weather condition at the time of the
        measurement.

    Raises
    ------
    requests.exceptions.HTTPError
        If the API replied with an HTTP error.

    """

    weather = get_weather(instrument, date, timeout)
    if weather is not None:
        frame = pandas.DataFrame(weather)
        frame.sort_values('measurement_time')
        return frame

    return None
