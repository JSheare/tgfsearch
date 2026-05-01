"""Parameters for use by the TGF search program and its modules."""

"""General Parameters"""
SEC_PER_DAY = 86400  # Number of seconds in a day
SEC_PER_HOUR = 3600  # Number of seconds in an hour
CENTURY = '20'  # The current century (numerically)
TWO_AM = 7200  # Number of seconds of the day corresponding to 2:00AM
WEATHER_PADDING = 3  # The number of hours on either side of an event to check the weather for
ABS_MEMORY_ALLOWANCE = 8589934592  # The upper bound of memory that the program can use for data in bytes
MEMORY_ALLOWANCE_FRAC = 0.40  # The fraction of available memory on the system that the program can use for data.
#   Bounded by the ABS_MEMORY_ALLOWANCE


"""Trace-Related Parameters"""
LARGE_TRIGSPOT = 4092  # The usual trigspot for the large buffer
SMALL_TRIGSPOT = 1024  # The usual trigspot for the smaller buffers
NUM_BINS_BASELINE = 100  # The number of bins from the very beginning of the trace to use when determining the baseline
TRIGGER_ABOVE_BASELINE = 2  # The number of energy channels above baseline needed to trigger a trace
MIN_RISING_EDGE_BINS = 71  # The minimum number of bins in a rising edge for it to be considered valid
ABOVE_BASELINE_RATIO_THRESH = 0.1  # Traces with no saturation and a ratio of above/below baseline counts > 1 must
#   have a ratio at least this much greater than 1 to be considered passing
BELOW_BASELINE_RATIO_THRESH = 0.5  # Traces with no saturation and a ratio of above/below baseline counts < 1 must
#   have a ratio at least this much greater than 0 to be considered passing
ROLLOVER_PERIOD = 2**36  # Maximum number of clock ticks before a rollover
T_STEP = 12.5e-9  # clock tick time resolution (80 MHz)
DT = 200e-9  # Time to extend sample on either side to let pulse shapes finish
TRACE_TRIGGER_THRESH = 5  # Threshold in mV for triggering a count from a trace
MV_PER_ADC = 0.2442002442002442  # Conversion for ADC scale (fixed)
DEADTIME_I = 24  # Holdoff (dead) time (units of samples)
INT_I = 24  # Integration time (units of samples)
PARTIAL_INT_I = 12  # Integration time (units of samples) for PSD partial integration
ENERGY_RESCALE = 1.0  # Possible energy rescaling for trace to counts
DEADTIME_EXTEND = 1  # Number of samples to extend deadtime by if not yet back to baseline after holdoff


"""Short Event Search Parameters"""
# Search algorithm parameters
INDIV_ROLLGAP = 4  # The rollgap used for individual scintillator searches (onescint, allscints)
ROLLGAP_CONTRIBUTION = 3.125  # The average rollgap contribution for each scintillator after the first one
AIRCRAFT_ROLLGAP = 18  # The rollgap used in aircraft mode
SHORT_EVENT_TIME_SPACING = 1e-3  # 1 millisecond
SHORT_EVENT_MIN_COUNTS = 10  # The minimum number of counts that a short event needs to be
MAX_PLOTS_PER_SCINT = 1000  # The maximum number of scatter plots/event files that can be made per scintillator

# Filter/ranking parameters
GOOD_LEN_THRESH = 30  # The number of counts at which a short event becomes *definitely* interesting

#   Noise filter parameters
CHANNEL_RANGE_WIDTH = 300  # The width of channels examined by the low/high energy ratio filter
CHANNEL_SEPARATION = 100  # The number of channels separating the low/high channel ranges
LOW_CHANNEL_START = 200  # The starting channel of the low energy channel range
CHANNEL_RATIO = 0.5  # The high/low energy channel range ratio required by the high/low energy ratio filter
MIN_NOISE_COUNTS = 3  # The minimum number of non-noise counts required for an event
NOISE_CUTOFF_ENERGY = 300  # The threshold for what's considered a noise/non-noise count

#   Successive CRS filter/clumpiness parameters
DIFFERENCE_THRESH = 2e-6  # The maximum time separation between two counts in a single clump
GAP_THRESH = 10e-6  # The minimum time separation between two counts of two different clumps
CLUMPINESS_THRESH = 0.27  # The clumpiness above which an event is probably just a successive CRS
CLUMPINESS_TOSSUP = 0.2  # The clumpiness at which an event could either be successive CRS or a real event

#   High energy lead parameters
HIGH_ENERGY_LEAD_THRESH = 15000  # The cutoff for what is considered a high-energy count

#   Ranking weight parameters (should add up to 1)
LEN_WEIGHT = 0.3
CLUMPINESS_WEIGHT = 0.2
HEL_WEIGHT = 0.2
WEATHER_WEIGHT = 0.3

# Scatter plot formatting parameters
SE_TIMESCALE_ONE = 5e-4  # Subplot 1 timescale, 500 Microseconds
SE_TIMESCALE_TWO = 0.005  # Subplot 2 timescale, 5 milliseconds
SE_TIMESCALE_THREE = 2  # Subplot 3 timescale, 2 seconds
DOT_ALPHA = 0.5  # Controls dot transparency


"""Long Event Search Parameters"""
# Search algorithm parameters
LPL_CHANNEL_CUTOFF = 2000  # All large plastic channels below this are cut during the long event search
NAI_CHANNEL_CUTOFF = 6000  # All sodium iodide channels below this are cut during the long event search
SHORT_BIN_SIZE = 4  # The size of each short bin (in seconds)
LONG_BIN_SIZE = 60  # The size of each long bin (in seconds)
FLAG_THRESH = 5  # The number of standard deviations above the mean at which a bin is flagged
LONG_EVENT_MIN_COUNTS = 1000  # Only in aircraft mode

# Normal baseline parameters
N_WINDOW_SIZE = 21600  # The size of the window to be used in the savgol filter (in seconds)
POLY_ORDER = 3  # The order of the polynomial to be used in the savgol filter

# Aircraft baseline parameters
A_WINDOW_SIZE = 80  # The (approximate) number of seconds in the window on each side
A_WINDOW_GAP = 20  # The (approximate) number of seconds between the center bin and the start of the window on each side

# Histogram subplot formatting parameters
LE_MAIN_BAR_COLOR = 'r'  # The color of the bars on the whole-day subplot
LE_MAIN_BAR_ALPHA = 0.5  # Controls the transparency of the bars in the whole-day subplot
LE_THRESH_LINE_COLOR = 'blue'  # Color of line representing triggering threshold
LE_SUBPLOT_PADDING = 100  # The max number of bins to pad a long event by on either side
LE_SUBPLOT_BAR_COLOR = 'c'  # Color of histogram bars in the subplots
LE_SUBPLOT_BAR_ALPHA = 0.5  # Controls the transparency of the bars in the subplots
