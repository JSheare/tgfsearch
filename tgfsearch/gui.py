"""A graphical user interface for running the TGF search program."""
import multiprocessing as multiprocessing
import os as os
import sys as sys
import threading as threading
import time as time
import tkinter as tk
import traceback as traceback
from queue import Queue
from tkinter import filedialog
from tkinter import ttk

# Adds parent directory to sys.path. Necessary to make the imports below work when running this file as a script
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import tgfsearch.tools.tools as tl
from tgfsearch.search import search_check, program


# Redirects stdout and stderr from the search program. Meant to be run in a subprocess
def search_program_wrapper(write, first_date, second_date, unit, mode_info):
    # For running the program with pythonw (no terminal)
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')

    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

    # Redirecting stdout into the pipe
    old_stdout_write = sys.stdout.write
    sys.stdout.write = write.send
    try:
        program(first_date, second_date, unit, mode_info)
    except Exception as ex:
        print('Error: search program terminated with the following error or warning:\n')
        # Removing the top layer of the traceback (which is just this function) and printing the remainder
        count = len(traceback.extract_tb(ex.__traceback__)) - 1
        print(traceback.format_exc(limit=-count))

    sys.stdout.write = old_stdout_write


# A helper class that keeps track of the required arguments for a single search
class SearchArgs:
    def __init__(self, first_date, second_date, detector, mode_info):
        self.first_date = first_date
        self.second_date = second_date
        self.detector = detector
        self.mode_info = mode_info

    def __str__(self):
        search_string = f'{self.first_date} {self.second_date} {self.detector}'
        for arg in self.mode_info:
            search_string += f' {arg}'

        return search_string

    def __hash__(self):
        return hash(tuple([self.first_date, self.second_date, self.detector] + self.mode_info))

    def __eq__(self, args2):
        return (self.first_date == args2.first_date and
                self.second_date == args2.second_date and
                self.detector == args2.detector and
                self.mode_info == args2.mode_info)


# A class for managing the search and keeping track of all the enqueued search information
class SearchManager:
    def __init__(self, write_func=print):
        self._write = write_func  # The function for writing output

        self._mode_flags = {}
        self._search_queue = Queue()  # Queue that holds all the enqueued searches
        self._search_set = set()  # Set that keeps track of already-enqueued searches so that no duplicates are added

        self._lock = threading.Lock()  # Mutex to prevent race conditions with a search in progress
        self._running = threading.Event()  # Event that marks an active search
        self._stop_event = threading.Event()  # Event for stopping the search

    # Enables the given mode
    def add_mode(self, mode):
        with self._lock:
            self._mode_flags[mode] = True

    # Disables the given mode
    def remove_mode(self, mode):
        with self._lock:
            if mode in self._mode_flags:
                self._mode_flags[mode] = False

    # Returns the number of searches in the queue
    def size(self):
        return self._search_queue.qsize()

    # Enqueues a new search with the given parameters
    def enqueue(self, first_date, second_date, detector, import_loc, export_loc):
        with self._lock:
            if second_date == 'yymmdd' or second_date == '':
                second_date = first_date

            # If the search command is valid, sets up a SearchArgs object to store it
            check = search_check(first_date, second_date, detector)
            if not(check[0]):
                self._write(check[1])
            else:
                mode_info = []
                for mode in self._mode_flags:
                    if self._mode_flags[mode]:
                        mode_info.append(mode)

                mode_info.append('-c')
                if import_loc == '':
                    import_loc = 'none'

                if export_loc == '':
                    export_loc = 'none'

                mode_info.append(import_loc)
                mode_info.append(export_loc)
                search_args = SearchArgs(first_date, second_date, detector.upper(), mode_info)
                # Enqueues the search if it isn't a duplicate
                if search_args not in self._search_set:
                    # Reasoning behind 3: one for custom, the last two for custom import/export locations
                    modes_string = f' [{", ".join(mode_info[0:-3]).replace("-", "")}]' if len(mode_info) > 3 else ''
                    self._write(f'Enqueueing {tl.short_to_full_date(first_date)}'
                          f'{" - " + tl.short_to_full_date(second_date) if first_date != second_date else ""}'
                          f' on {detector.upper()}{modes_string}.')
                    self._search_queue.put(search_args)
                    self._search_set.add(search_args)

    # Runs all the enqueued searches
    def run(self):
        with self._lock:
            self._running.set()
            while not self._search_queue.empty():
                search_args = self._search_queue.get()
                self._search_set.remove(search_args)

                # Outputs feedback about what date and modes were selected
                feedback_string = f'\nRunning search for {tl.short_to_full_date(search_args.first_date)}'
                if search_args.first_date != search_args.second_date:
                    feedback_string += f' - {tl.short_to_full_date(search_args.second_date)}'

                feedback_string += f' on {search_args.detector}.'
                self._write(feedback_string)
                # Reasoning behind 3: one for custom, the last two for custom import/export locations
                if len(search_args.mode_info) > 3:
                    self._write(f'This search will be run with the following modes: '
                                 f'{", ".join(search_args.mode_info[0:-3]).replace("-", "")}.')

                # Runs the search program in a separate process and manages it
                read, write = multiprocessing.Pipe()
                process = multiprocessing.Process(target=search_program_wrapper,
                                                  args=(write, search_args.first_date, search_args.second_date,
                                                        search_args.detector, search_args.mode_info))
                process.start()
                while process.is_alive() and not self._stop_event.is_set():
                    # Writes the processes' piped stdout
                    while read.poll():
                        self._write(read.recv(), end='')

                    # Waiting a little while before checking for more
                    time.sleep(0.20)

                if process.is_alive():  # This will be executed if stop() is run
                    process.terminate()
                    process.join()
                    break

                # In case there's still some strings left in the pipe
                while read.poll():
                    self._write(read.recv(), end='')

            self._write('\nSearch Concluded.\n')
            self._stop_event.clear()
            self._running.clear()

    # Stops the search if it's running
    def stop(self):
        if self._running.is_set():
            self._stop_event.set()

    # Clears the search queue and resets the selected modes
    def reset(self):
        with self._lock:
            while not self._search_queue.empty():
                self._search_queue.get()

            self._search_set.clear()
            for mode in self._mode_flags:
                self._mode_flags[mode] = False


# A class implementing the search GUI window, all its widgets, and their associated functionality
class SearchWindow(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._master = master
        self._checkbox_variables = []
        self._toggleable_widgets = []

        # Making and placing the text box display
        self._text_box = tk.Text(self, height=30, width=100)
        self._text_box['state'] = tk.DISABLED
        self._text_box.grid(row=0, column=1, columnspan=3)

        self._text_box_label = tk.Label(self, text='Search Output')
        self._text_box_label.grid(row=1, column=2, pady=(5, 0))

        # Setting up the input frame
        self._input_frame = tk.Frame(self)
        self._input_frame.grid(row=2, column=1, rowspan=2)

        # Adding and placing the date/detector labels and entry boxes to the frame
        self._date_one_label = tk.Label(self._input_frame, text='Date One:')
        self._date_one_label.grid(row=0, column=0, pady=(5, 0))

        self._date_one_entry = tk.Entry(self._input_frame, width=15, borderwidth=5)
        self._date_one_entry.insert(0, 'yymmdd')
        self._date_one_entry.bind('<FocusIn>', lambda e: self._clear_ghost_text(self._date_one_entry, 'yymmdd'))
        self._date_one_entry.grid(row=1, column=0, pady=(5, 0))
        self._toggleable_widgets.append(self._date_one_entry)

        self._date_two_label = tk.Label(self._input_frame, text='Date Two:')
        self._date_two_label.grid(row=2, column=0, pady=(5, 0))

        self._date_two_entry = tk.Entry(self._input_frame, width=15, borderwidth=5)
        self._date_two_entry.insert(0, 'yymmdd')
        self._date_two_entry.bind('<FocusIn>', lambda e: self._clear_ghost_text(self._date_two_entry, 'yymmdd'))
        self._date_two_entry.grid(row=3, column=0, pady=(5, 0))
        self._toggleable_widgets.append(self._date_two_entry)

        self._detector_label = tk.Label(self._input_frame, text='Detector:')
        self._detector_label.grid(row=4, column=0, pady=(5, 0))

        self._detector_entry = tk.Entry(self._input_frame, width=15, borderwidth=5)
        self._detector_entry.grid(row=5, column=0, pady=(5, 0))
        self._toggleable_widgets.append(self._detector_entry)

        # Setting up the search control frame
        self._search_frame = tk.Frame(self)
        self._search_frame.grid(row=2, column=2, rowspan=2)

        # Adding and placing the start, enqueue, and stop buttons, and the enqueue counter
        self._start_button = tk.Button(self._search_frame, height=3, width=20, text='Start', bg='white',
                                       command=self._start)
        self._start_button.grid(row=0, column=0, columnspan=2, pady=(5, 0))
        self._toggleable_widgets.append(self._start_button)

        self._enqueue_button = tk.Button(self._search_frame, height=3, width=8, text='Enqueue', bg='white',
                                         command=self._enqueue)
        self._enqueue_button.grid(row=1, column=0, pady=(5, 0))
        self._toggleable_widgets.append(self._enqueue_button)

        self._stop_button = tk.Button(self._search_frame, height=3, width=8, text='Stop', bg='white',
                                      command=self._stop)
        self._stop_button.grid(row=1, column=1, pady=(5, 0))

        self._enqueue_label = tk.Label(self._search_frame, text='')
        self._enqueue_label.grid(row=2, column=0, columnspan=2, pady=(5, 0))

        # Setting up the display frame
        self._display_frame = tk.Frame(self)
        self._display_frame.grid(row=2, column=3)

        # Adding and placing the clear text and reset/ clear queue buttons
        self._clear_button = tk.Button(self._display_frame, height=3, width=8, text='Clear\nText', bg='white',
                                       command=self._clear)
        self._clear_button.grid(row=0, column=0, padx=(0, 4), pady=(5, 0))

        self._reset_button = tk.Button(self._display_frame, height=3, width=8, text='Reset/\nClear\nQueue', bg='white',
                                       command=self._reset)
        self._reset_button.grid(row=0, column=1, padx=(4, 0), pady=(5, 0))

        # Setting up the modes frame
        self._modes_frame = tk.Frame(self)
        self._modes_frame.grid(row=3, column=3, pady=(5, 0))

        # Adding and placing the mode label and checkboxes
        self._cb_label = tk.Label(self._modes_frame, text='Modes:')
        self._cb_label.grid(row=0, column=0, columnspan=2, pady=(5, 0))

        oscb = tk.IntVar()
        self._onescint_cb = tk.Checkbutton(self._modes_frame, text='onescint', variable=oscb, onvalue=1, offvalue=0,
                                           command=lambda: self._check_uncheck(oscb, '--onescint'))
        self._onescint_cb.grid(row=1, column=0, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(oscb)
        self._toggleable_widgets.append(self._onescint_cb)

        ascb = tk.IntVar()
        self._allscints_cb = tk.Checkbutton(self._modes_frame, text='allscints', variable=ascb, onvalue=1, offvalue=0,
                                            command=lambda: self._check_uncheck(ascb, '--allscints'))
        self._allscints_cb.grid(row=2, column=0, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(ascb)
        self._toggleable_widgets.append(self._allscints_cb)

        acb = tk.IntVar()
        self._aircraft_cb = tk.Checkbutton(self._modes_frame, text='aircraft', variable=acb, onvalue=1, offvalue=0,
                                           command=lambda: self._check_uncheck(acb, '--aircraft'))
        self._aircraft_cb.grid(row=3, column=0, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(acb)
        self._toggleable_widgets.append(self._aircraft_cb)

        cecb = tk.IntVar()
        self._clnenrg_cb = tk.Checkbutton(self._modes_frame, text='clnenrg', variable=cecb, onvalue=1, offvalue=0,
                                          command=lambda: self._check_uncheck(cecb, '--clnenrg'))
        self._clnenrg_cb.grid(row=4, column=0, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(cecb)
        self._toggleable_widgets.append(self._clnenrg_cb)

        stcb = tk.IntVar()
        self._sktrace_cb = tk.Checkbutton(self._modes_frame, text='sktrace', variable=stcb, onvalue=1, offvalue=0,
                                          command=lambda: self._check_uncheck(stcb, '--sktrace'))
        self._sktrace_cb.grid(row=1, column=1, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(stcb)
        self._toggleable_widgets.append(self._sktrace_cb)

        sscb = tk.IntVar()
        self._skshort_cb = tk.Checkbutton(self._modes_frame, text='skshort', variable=sscb, onvalue=1, offvalue=0,
                                          command=lambda: self._check_uncheck(sscb, '--skshort'))
        self._skshort_cb.grid(row=2, column=1, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(sscb)
        self._toggleable_widgets.append(self._skshort_cb)

        sgcb = tk.IntVar()
        self._skglow_cb = tk.Checkbutton(self._modes_frame, text='skglow', variable=sgcb, onvalue=1, offvalue=0,
                                         command=lambda: self._check_uncheck(sgcb, '--skglow'))
        self._skglow_cb.grid(row=3, column=1, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(sgcb)
        self._toggleable_widgets.append(self._skglow_cb)

        pcb = tk.IntVar()
        self._pickle_cb = tk.Checkbutton(self._modes_frame, text='pickle', variable=pcb, onvalue=1, offvalue=0,
                                         command=lambda: self._check_uncheck(pcb, '--pickle'))
        self._pickle_cb.grid(row=4, column=1, sticky=tk.W, pady=(3, 0))
        self._checkbox_variables.append(pcb)
        self._toggleable_widgets.append(self._pickle_cb)

        # Setting up the file import/export frame
        self._file_frame = tk.Frame(self)
        self._file_frame.grid(row=5, column=1, columnspan=3, pady=(10, 0))
        self._file_frame.columnconfigure(3, {'minsize': 30})

        # Adding and placing the custom import label, entry box, and file dialogue button
        self._import_label = tk.Label(self._file_frame, text='Import Location:')
        self._import_label.grid(row=0, column=0, columnspan=2, pady=(5, 0))

        self._import_entry = tk.Entry(self._file_frame, width=40, borderwidth=5)
        self._import_entry.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        self._toggleable_widgets.append(self._import_entry)

        self._import_button = tk.Button(self._file_frame, width=6, height=2, text='Browse',
                                        command=lambda: self._select_dir(self._import_entry))
        self._import_button.grid(row=1, column=2, pady=(5, 0))
        self._toggleable_widgets.append(self._import_button)

        # Adding and placing the custom export label, entry box, and file dialogue button
        self._export_label = tk.Label(self._file_frame, text='Export Location:')
        self._export_label.grid(row=0, column=4, columnspan=2, pady=(5, 0))

        self._export_entry = tk.Entry(self._file_frame, width=40, borderwidth=5)
        self._export_entry.grid(row=1, column=4, columnspan=2, pady=(5, 0))
        self._toggleable_widgets.append(self._export_entry)

        self._export_button = tk.Button(self._file_frame, width=6, height=2, text='Browse',
                                        command=lambda: self._select_dir(self._export_entry))
        self._export_button.grid(row=1, column=6, pady=(5, 0))
        self._toggleable_widgets.append(self._export_button)

        # Modes separator line
        ttk.Separator(self, orient='horizontal').place(in_=self._modes_frame, bordermode='outside', y=3, relwidth=1.0)
        # Import/export separator line
        ttk.Separator(self, orient='horizontal').place(in_=self._file_frame, bordermode='outside', relwidth=1.0)

        # Setting up the search manager
        self._write_queue = Queue()
        self._search_manager = SearchManager(self.write)
        self._search_thread = None

        # Setting up events that modify the state of the window
        self.bind('<<enable_widgets>>', self._enable_widget_handler)
        self.bind('<<disable_widgets>>', self._disable_widget_handler)
        self.bind('<<write>>', self._write_handler)

        # Starting the enqueue counter updater
        self._enqueued_counter_interval = 20  # interval at which search queue size is checked in milliseconds
        self._update_enqueued_counter()

    # Enqueues strings for writing to the big text box and notifies the handler
    def write(self, *args, sep=' ', end='\n', **kwargs):
        if len(args) > 0:
            output = sep.join(args) + end
        else:
            output = args[0] + end

        self._write_queue.put(output)
        self.event_generate('<<write>>', when='tail')

    # Writes text from the output queue to the big text box when the write event happens
    def _write_handler(self, event):
        self._text_box['state'] = tk.NORMAL
        while not self._write_queue.empty():
            self._text_box.insert('end', self._write_queue.get(), 'last_insert')
            self._text_box.yview(tk.END)

        self._text_box['state'] = tk.DISABLED

    # Creates a file dialogue and then puts the selected directory into the specified text entry box
    @staticmethod
    def _select_dir(entry_box):
        directory = filedialog.askdirectory(initialdir='/')
        entry_box.delete(0, 'end')
        entry_box.insert(0, directory)

    # Clears the sample text from the given text entry box
    @staticmethod
    def _clear_ghost_text(entry_box, ghost_text):
        current_text = entry_box.get()
        if current_text == ghost_text:
            entry_box.delete(0, 'end')

    # Updates the enqueued searches counter periodically
    def _update_enqueued_counter(self):
        self._enqueue_label['text'] = f'Searches\nEnqueued:\n{self._search_manager.size()}'
        self.after(self._enqueued_counter_interval, self._update_enqueued_counter)

    # Adds/removes the given mode from the search arguments when the corresponding checkbox is checked/unchecked
    def _check_uncheck(self, var, mode):
        if var.get() == 1:
            self._search_manager.add_mode(mode)
        else:
            self._search_manager.remove_mode(mode)

    # Enables/disables GUI widgets depending on the action parameter
    def _change_widgets(self, action):
        for widget in self._toggleable_widgets:
            widget['state'] = action

    # Enables widgets when the enable widget event is received
    def _enable_widget_handler(self, event):
        self._change_widgets(tk.NORMAL)

    # Disables widgets when the enable widget event is received
    def _disable_widget_handler(self, event):
        self._change_widgets(tk.DISABLED)

    # Enqueues a new search based on the current contents of all the text entry boxes
    def _enqueue(self):
        if self._search_thread is None or not self._search_thread.is_alive():
            self._search_manager.enqueue(self._date_one_entry.get(), self._date_two_entry.get(),
                                         self._detector_entry.get(), self._import_entry.get(), self._export_entry.get())

    # Starts running the enqueued searches
    def _start(self):
        self._enqueue()  # In case the current info hasn't been enqueued yet
        if (self._search_thread is None or not self._search_thread.is_alive()) and self._search_manager.size() > 0:
            self._search_thread = threading.Thread(target=self._run, args=())
            self._search_thread.start()  # Running the search in another thread to prevent the GUI from locking up

    # Target function for search thread. Disables/Enables the GUI elements while the searches are running
    def _run(self):
        self.event_generate('<<disable_widgets>>', when='tail')
        self._search_manager.run()
        self.event_generate('<<enable_widgets>>', when='tail')

    # Stops the search if there is one
    def _stop(self):
        if self._search_thread is not None and self._search_thread.is_alive():
            self._search_manager.stop()

    # Clears the big text box
    def _clear(self):
        self._text_box['state'] = tk.NORMAL
        self._text_box.delete('1.0', 'end')
        self._text_box['state'] = tk.DISABLED

    # Stops the search and resets the GUI widgets back to their default states
    def _reset(self):
        self._stop()
        self._search_manager.reset()  # Clearing the search queue and mode flags
        self._clear()

        self._date_one_entry.delete(0, 'end')
        self._date_one_entry.insert(0, 'yymmdd')
        self._date_two_entry.delete(0, 'end')
        self._date_two_entry.insert(0, 'yymmdd')
        self._detector_entry.delete(0, 'end')

        self._import_entry.delete(0, 'end')
        self._export_entry.delete(0, 'end')

        # Unchecking the checkboxes
        for variable in self._checkbox_variables:
            variable.set(0)


def main():
    # For running the program with pythonw (no terminal)
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')

    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

    root = tk.Tk()
    root.title('TGF Search')

    gui = SearchWindow(root)
    gui.pack(padx=130, pady=10)
    root.mainloop()


if __name__ == '__main__':
    main()
