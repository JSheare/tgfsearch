"""A module containing a subclass of Instrument for handling instruments with arbitrary scintillator configurations."""
from __future__ import annotations

import os as os

import tgfsearch.helpers.helper_funcs as helper_funcs
from tgfsearch.instruments.instrument import Instrument
from tgfsearch.instruments.scintillator import Scintillator


class AdaptiveInstrument(Instrument):
    """A subclass of Instrument made to handle data for instruments with arbitrary scintillator configurations."""
    def __init__(self, date_str: str) -> None:
        super().__init__('ADAPTIVE', date_str, read_identity=False)

    def _reset_identity(self):
        """Resets everything back to its default state (no identity)."""
        self.clear()
        self._has_identity = False
        self.name = 'ADAPTIVE'
        self._scintillators.clear()
        self._scint_list.clear()
        self.deployment = self._get_deployment()

    def _infer_identity(self) -> None:
        """Uses files in import loc to determine instrument type and scintillator configuration."""
        all_files = helper_funcs.get_serial_num_filelist(self._import_loc, '*', self.date)
        if len(all_files) == 0:
            raise FileNotFoundError('no data files to infer instrument configuration from.')

        # Attempting to infer instrument name based on the names of parent directories
        parent_dirs = all_files[0].replace('\\', '/').split('/')[:-1]
        if len(parent_dirs) >= 3:
            self.name = parent_dirs[-3].upper()
            # Regenerating deployment info and results directory based on the new name
            self.deployment = self._get_deployment()

        try:
            entries = helper_funcs.read_json_file(
                f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config/instrument_config.json')

        except SyntaxError:
            raise SyntaxError('invalid syntax in instrument config file.')

        # Getting all supported scintillators and their priorities (greatest to least)
        scintillator_priority = entries['scintillator_priority']

        # Attempting to infer scintillator configuration based on the data files present
        # Walking each file to determine its corresponding scintillator
        scintillator_numbers = {}
        for file in all_files:
            index = 0
            for i in range(len(file) - 1, 0, -1):
                if file[i] == '/' or file[i] == '\\':
                    break

                index = i

            # Recording the scintillator name and eRC number if it hasn't been seen yet
            scintillator = file[index + 7: index + 10]
            if scintillator not in scintillator_priority:
                raise ValueError('unknown or unsupported scintillator type')

            if scintillator not in scintillator_numbers:
                scintillator_numbers[scintillator] = file[index + 3: index + 7]  # Recording the eRC number
                self._scint_list.append(scintillator)

        # Putting the scintillators in the correct order of priority
        self._scint_list.sort(key=lambda scint: scintillator_priority.index(scint))

        # Making the scintillator objects
        for scintillator in self._scint_list:
            self._scintillators[scintillator] = Scintillator(scintillator, scintillator_numbers[scintillator],
                                                             self.date, 'thor_lm')

        self._has_identity = True

    def set_import_loc(self, loc: str) -> None:
        """Sets the directory where data will be imported from. WARNING: calling this will also reset the Instrument's
        current identity and infer a new one from the passed import loc.

        Parameters
        ----------
        loc : str
            The import directory as a string.

        """

        self._reset_identity()
        super().set_import_loc(loc)
        self._infer_identity()

    def get_clone(self) -> AdaptiveInstrument:
        """Returns a new Instrument with the same identity as the current one (but no data).

        Returns
        -------
        tgfsearch.instruments.instrument.AdaptiveInstrument
            A new Instrument with the same identity as the current one. Identity includes: instrument name, date,
            deployment info, import directory, and scintillator configuration.

        """

        clone = type(self)(helper_funcs.date_to_yymmdd(self.date))
        if self.has_identity:
            clone._import_loc = self._import_loc
            clone._infer_identity()

        return clone
