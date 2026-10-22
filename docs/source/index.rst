.. tgfsearch documentation master file, created by
   sphinx-quickstart on Wed Oct 21 16:00:06 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tgfsearch documentation
=======================

A package containing tools and scripts for searching raw data from the instruments of the TGF group at UCSC.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Data Reader
===========

Functions for reading UCSC TGF group data files.

.. automodule:: tgfsearch.tools.data_reader
   :members:


Reader Class
============

A class that wraps and serves as an easy interface for the data reader.

.. automodule:: tgfsearch.tools.reader
   :members:
   

Instrument Class
================

A class for keeping track of instrument data and associated information.

.. automodule:: tgfsearch.instruments.instrument
   :members:


Data Tools
==========

Tools for working with UCSC TGF group data.

.. automodule:: tgfsearch.tools.tools
   :members:


API
===

Functions used to access the UCSC TGF group API.
   
.. automodule:: tgfsearch.helpers.api
   :members:
	


