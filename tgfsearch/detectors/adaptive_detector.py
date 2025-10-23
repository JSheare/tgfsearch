"""Child class of Detector made to handle data for instruments with arbitrary scintillator configurations."""
import json as json
import os as os


from tgfsearch.detectors.detector import Detector
from tgfsearch.detectors.scintillator import Scintillator


class AdaptiveDetector(Detector):
    def __init__(self, date_str):
        super().__init__('ADAPTIVE', date_str, read_identity=False)

    def _reset_identity(self):
        """Resets everything back to its default state (no identity)."""
        self.clear()
        self._has_identity = False
        self._results_loc = self._results_loc.replace(f'Results/{self.unit}', 'Results/ADAPTIVE')
        self.unit = 'ADAPTIVE'
        self._scintillators.clear()
        self.scint_list.clear()
        self.deployment = self._get_deployment()

    def _infer_identity(self):
        """Uses files in import loc to determine instrument type and scintillator configuration."""
        all_files = self._get_serial_num_filelist('*')
        if len(all_files) == 0:
            raise FileNotFoundError('no data files to infer instrument configuration from.')

        # Attempting to infer instrument name based on the names of parent directories
        parent_dirs = all_files[0].replace('\\', '/').split('/')[:-1]
        if len(parent_dirs) >= 3:
            self.unit = parent_dirs[-3].upper()
            # Regenerating deployment info and results directory with the new name
            self.deployment = self._get_deployment()
            self._results_loc = self._results_loc.replace('Results/ADAPTIVE', f'Results/{self.unit}')

        # Getting the default growth factors (using Thor format because it's the most likely)
        try:
            with open(f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config/detector_config.json',
                      'r') as config_file:
                entries = json.load(config_file)

                # Getting the default growth factors (using Thor format because it's the most likely)
                default_lm_growth = entries['growth_factors']['thor_lm']['lm_growth_factor']
                default_trace_growth = entries['growth_factors']['thor_lm']['lm_growth_factor']

                # Getting all supported scintillators and their priorities (greatest to least)
                scintillator_priority = entries['scintillator_priority']

        except json.decoder.JSONDecodeError:
            raise SyntaxError('invalid syntax in detector config file.')

        # Attempting to infer scintillator configuration based on the data files present
        # Walking each file to determine its corresponding scintillator
        for file in all_files:
            index = 0
            for i in range(len(file) - 1, 0, -1):
                if file[i] == '/' or file[i] == '\\':
                    break

                index = i

            # Making a new Scintillator if one doesn't exist already
            scintillator = file[index + 7: index + 10]
            if scintillator not in scintillator_priority:
                raise ValueError('unknown or unsupported scintillator type')

            if scintillator not in self._scintillators:
                eRC = file[index + 3: index + 7]
                self._scintillators[scintillator] = Scintillator(scintillator, eRC)
                self.lm_growth_factors[scintillator] = default_lm_growth
                self.trace_growth_factors[scintillator] = default_trace_growth
                self.scint_list.append(scintillator)

        # Assigning the default scintillator based on the above priority
        for scint in scintillator_priority:
            if scint in self._scintillators:
                self.default_scintillator = scint
                break

        self._has_identity = True

    def set_import_loc(self, loc):
        self._reset_identity()
        super().set_import_loc(loc)
        self._infer_identity()

    def get_clone(self):
        clone = type(self)(self.date_str)
        if self.has_identity:
            clone._import_loc = self._import_loc
            clone._results_loc = self._results_loc
            clone._infer_identity()

        return clone
