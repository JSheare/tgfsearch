"""A class that wraps and serves as an easy interface for the data reader."""

import tgfsearch.data_reader as dr


class Reader:
    """A class that wraps and serves as an easy interface for the data reader.

    Attributes
    ----------
    passtime : dict
        A dictionary containing information needed to import the subsequent list mode files properly (if applicable).

    """
    def __init__(self):
        self.passtime = dr.get_passtime()

    def __call__(self, file_name, reset_after=False):
        """Defines the behavior for calling a class instance like a function."""
        return self.read(file_name, reset_after)

    def reset(self):
        """Resets the reader instance back to its default state."""
        self.passtime = dr.get_passtime()

    def read(self, file_name, reset_after=False, clean_energy=False):
        """Reads the data file with the given name and returns the data as a pandas dataframe.

        Parameters
        ----------
        file_name : str
            The name of the data file to be read.

        reset_after : bool
            Optional. If True, resets the reader instance back to its default state after the data file has been
            read.
        clean_energy : bool
            Optional. If True, asks the data reader to strip out maximum and low energy counts.

        Returns
        -------
        pandas.core.frame.DataFrame
            A pandas dataframe containing the given file's data.

        """

        data, self.passtime = dr.read_file(file_name, self.passtime, killcr=clean_energy)
        if reset_after:
            self.reset()

        return data
