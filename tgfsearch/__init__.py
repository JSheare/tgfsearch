"""A package containing tools and scripts for searching raw data from the instruments of the TGF group at UCSC."""

# Everything here should be available at the top level of the package upon importing
from tgfsearch.tools.tools import *
from tgfsearch.tools.data_reader import get_passtime, read_file, read_files, translate_flags
from tgfsearch.instruments.adaptive_instrument import AdaptiveInstrument
from tgfsearch.instruments.instrument import Instrument
from tgfsearch.tools.reader import Reader
