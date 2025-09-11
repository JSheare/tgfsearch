"""A script that turns the package's various docstrings into HTML documents."""
import os
import pydoc
import shutil
import sys


def make_doc(name, parent_dir):
    sys.path.append(parent_dir)
    os.chdir(os.getcwd() + '/docs')
    pydoc.writedoc(name)
    os.chdir(os.path.dirname(os.getcwd()))


def main():
    docs_path = os.getcwd() + '/docs'
    if os.path.exists(docs_path):
        shutil.rmtree(docs_path)

    os.mkdir(docs_path)

    # Docs for tools
    make_doc('tools', os.getcwd() + '/tgfsearch')

    # Docs for detector
    make_doc('detector', os.getcwd() + '/tgfsearch/detectors')

    # Docs for scintillator
    make_doc('scintillator', os.getcwd() + '/tgfsearch/detectors')

    # Docs for reader
    make_doc('reader', os.getcwd() + '/tgfsearch/helpers')


if __name__ == '__main__':
    main()
