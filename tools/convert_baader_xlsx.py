"""
Convert the Baader filter measurement sheets into the CSV format used by the
transmission templates in ``transmission_fitter/data/Templates``.

Baader ships one .xlsx per filter with two columns, wavelength in nm and the
optical density OD = -log10(T). This script converts them to throughput,
T = 10**(-OD), and writes ``,Wavelength,Throughput`` CSVs next to the source
sheets, i.e. the same layout as the other template files in the repository.

The .xlsx files are read with the standard library (an xlsx is a zip of XML),
so no Excel engine needs to be installed.

Usage:
    python tools/convert_baader_xlsx.py
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

FILTERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'transmission_fitter', 'data', 'Templates', 'filters')

## Source sheet -> output CSV. The CSV names follow the band keys used in
## fitutils.BAND_TEMPLATES.
CONVERSIONS = {
    'Baader-Sloan-SDSS_u-Filter.xlsx': 'baader_sdss_u.csv',
    'Baader-Sloan-SDSS_g-Filter.xlsx': 'baader_sdss_g.csv',
    'Baader-Sloan-SDSS_r-Filter.xlsx': 'baader_sdss_r.csv',
    'Baader-Sloan-SDSS_i-Filter.xlsx': 'baader_sdss_i.csv',
    'Baader-UBVRI_V-Filter.xlsx': 'baader_bessel_v.csv',
}


def read_xlsx_columns(path):
    """
    Read the first worksheet of an .xlsx file without an Excel engine.

    Parameters:
    - path (str): Path to the .xlsx file.

    Returns:
    - title (str): The text of cell A1.
    - header (tuple): The text of cells A2 and B2.
    - rows (list): The (A, B) cell values of the remaining rows, as strings.
    """
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            shared = [''.join(t.text or '' for t in si.iter(NS + 't'))
                      for si in root.findall(NS + 'si')]

        sheets = sorted(n for n in archive.namelist()
                        if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
        root = ET.fromstring(archive.read(sheets[0]))

        rows = []
        for row in root.iter(NS + 'row'):
            cells = {}
            for cell in row.findall(NS + 'c'):
                value_node = cell.find(NS + 'v')
                value = None if value_node is None else value_node.text
                if cell.get('t') == 's' and value is not None:
                    value = shared[int(value)]
                cells[re.sub(r'\d', '', cell.get('r'))] = value
            rows.append((cells.get('A'), cells.get('B')))

    return rows[0][0], (rows[1][0], rows[1][1]), rows[2:]


def convert(xlsx_name, csv_name):
    """
    Convert one Baader sheet to a throughput CSV.

    Parameters:
    - xlsx_name (str): File name of the source sheet, inside FILTERS_DIR.
    - csv_name (str): File name of the CSV to write, inside FILTERS_DIR.

    Returns:
    - df (pandas.DataFrame): The table that was written.
    """
    xlsx_path = os.path.join(FILTERS_DIR, xlsx_name)
    title, header, rows = read_xlsx_columns(xlsx_path)

    if header[0] is None or 'nm' not in header[0] or header[1] != 'OD':
        raise ValueError('{}: unexpected header {}, expected wavelength in nm and OD.'.format(
            xlsx_name, header))

    wavelength = np.array([float(a) for a, b in rows if a not in (None, '') and b not in (None, '')])
    od = np.array([float(b) for a, b in rows if a not in (None, '') and b not in (None, '')])

    throughput = 10.0 ** (-od)
    if np.any(throughput > 1.0) or np.any(throughput < 0.0):
        raise ValueError('{}: throughput outside [0, 1] after converting OD.'.format(xlsx_name))

    order = np.argsort(wavelength)
    df = pd.DataFrame({'Wavelength': wavelength[order], 'Throughput': throughput[order]})
    df = df.drop_duplicates(subset='Wavelength').reset_index(drop=True)

    csv_path = os.path.join(FILTERS_DIR, csv_name)
    df.to_csv(csv_path, float_format='%.6g')

    ## Half-power points of the passband, i.e. the contiguous block of points
    ## above half peak that contains the peak. Some filters leak again beyond
    ## 1200 nm, which must not be counted as part of the passband.
    peak_idx = int(df['Throughput'].idxmax())
    peak = df['Throughput'].iloc[peak_idx]
    above = (df['Throughput'] > 0.5 * peak).values
    lo = peak_idx
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak_idx
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1

    print('{:34s} -> {:22s} {:.0f}-{:.0f} nm, peak T={:.4f} at {:.0f} nm, FWHM {:.0f}-{:.0f} nm'.format(
        title, csv_name, df['Wavelength'].min(), df['Wavelength'].max(),
        peak, df['Wavelength'].iloc[peak_idx],
        df['Wavelength'].iloc[lo], df['Wavelength'].iloc[hi]))
    return df


if __name__ == '__main__':
    for xlsx_name, csv_name in CONVERSIONS.items():
        convert(xlsx_name, csv_name)
