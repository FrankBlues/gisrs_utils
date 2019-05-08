# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:13:40 2019
从NASA https://atmcorr.gsfc.nasa.gov网站请求Landsat相关资料

"""
import json
import re
import requests


def read_json(filename):
    """ Read data from JSON file.

    Args:
        filename (str): Name of JSON file to be read.

    Returns:
        dict: Data stored in JSON file.

    """
    with open(filename, 'r') as file:
        return json.load(file)


def request_atmcorr(json_param):
    """从NASA https://atmcorr.gsfc.nasa.gov网站请求相关参数:
        大气透过率(Band average atmospheric transmission).
        大气向上辐射亮度(Effective bandpass upwelling radianc, W/m^2/sr/um).
        大气向下辐射亮度(Effective bandpass downwelling radiance, W/m^2/sr/um).

    Args:
        json_param (txt): Json file for the needed params, which are:
            year, month, day, hour, minute: The date and time;
            thelat, thelong: The scene location;
            profile_option:
                1: Use atmospheric profile for closest integer lat/long.
                2: Use interpolated atmospheric profile for given lat/long.
            stdatm_option: Upper atmospheric profile
                1: Use mid-latitude summer standard atmosphere.
                2: Use mid-latitude winter standard atmosphere.
            L57_option: Which satelite
                11: Output only atmospheric profile, do not calculate
                    effective radiances.
                5: Use Landsat-5 Band 6 spectral response curve.
                7: Use Landsat-7 Band 6 spectral response curve .
                8: Use Landsat-8 TIRS Band 10 spectral response curve .
            altitude(km), pressure(mb), temperature(C), rel_humid(%):
                Surface Conditions.
                Optional, if you do enter surface conditions, all four
                conditions must be entered.
            user_email: User email.

    Returns:
        requests.models.Response: The response.

    """
    url = 'https://atmcorr.gsfc.nasa.gov/cgi-bin/atm_corr.pl'
    payload = read_json(json_param)
    res = requests.post(url, data=payload)
    if res.status_code == 200:
        return res
    return


def parse_atmcorr_res(res):
    if not isinstance(res, requests.models.Response):
        raise ValueError('Not a valid requests response.')

    matched = re.search(r'<br>Band average atmospheric transmission.*<br>',
                        res.text).group(0)
    if matched == '':
        raise ValueError('No match, check the request.')
    for i in matched.split('<br>'):
        if 'Band average atmospheric transmission' in i:
            trans = float(i.split(':')[1].strip())
        elif 'Effective bandpass upwelling radiance' in i:
            up_rad = float(i.split(':')[1].strip()[:4])
        elif 'Effective bandpass downwelling radiance' in i:
            down_rad = float(i.split(':')[1].strip()[:4])
    return trans, up_rad, down_rad


if __name__ == '__main__':
    f = 'd:/landsat_atmcorr_form.json'
    r = request_atmcorr(f)
    trans, up_rad, down_rad = parse_atmcorr_res(r)
    # parser = MyHTMLParser()
    # parser.feed(r.text)