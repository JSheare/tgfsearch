"""A script that turns the package's various docstrings into HTML documents."""
import os
import pydoc
import shutil
import sys


def make_doc(name, parent_dir, docs_path):
    sys.path.append(parent_dir)
    os.chdir(docs_path)
    pydoc.writedoc(name)
    os.chdir(os.path.dirname(os.getcwd()))

def make_index(docs_path):
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '\t<title>tgfsearch docs</title>\n'
        '</head?\n'
        '\n'
        '<body>\n'
        '\t<h1>tgfsearch Docs</h1>\n'
        '\t<br>\n'
        '\t<h2>Documentation for the major tgfsearch modules can be found below:</h2>\n'
        '\t<br>\n'
        '\t<a href="data_reader.html"> &#x2022; Data Reader Documentation</a>\n'
        '\t<br>\n'
        '\t<a href="detector.html"> &#x2022; Detector Documentation</a>\n'
        '\t<br>\n'
        '\t<a href="reader.html"> &#x2022; Reader Helper Documentation</a>\n'
        '\t<br>\n'
        '\t<a href="scintillator.html"> &#x2022; Scintillator Documentation</a>\n'
        '\t<br>\n'
        '\t<a href="tools.html"> &#x2022; Tools Documentation</a>\n'
        '</body>'
    )
    with open(f'{docs_path}/index.html', 'w') as file:
        file.write(html)


def main():
    docs_path = os.getcwd() + '/docs'
    if os.path.exists(docs_path):
        shutil.rmtree(docs_path)

    os.mkdir(docs_path)

    # Docs for tools
    make_doc('tools', os.getcwd() + '/tgfsearch', docs_path)

    # Docs for detector
    make_doc('detector', os.getcwd() + '/tgfsearch/detectors', docs_path)

    # Docs for scintillator
    make_doc('scintillator', os.getcwd() + '/tgfsearch/detectors', docs_path)

    # Docs for reader helper
    make_doc('reader', os.getcwd() + '/tgfsearch/helpers', docs_path)

    # Docs for data reader
    make_doc('data_reader', os.getcwd() + '/tgfsearch', docs_path)

    # Writing the index
    # Remember to update make_index if you add a new file.
    make_index(docs_path)


if __name__ == '__main__':
    main()
