# -*- coding: utf-8 -*-
"""Module with some useful time/date functions.
"""

import datetime


def get_dates_in_range(start_date, end_date):
    """Get all dates within input start and end date in ISO 8601 format.

    Args:
        start_date (str): Start date in ISO 8601 format.
        end_date (str): End date in ISO 8601 format.

    Returns:
        list: Dates between start_date and end_date in ISO 8601 format.

    """
    start_dt = iso_to_datetime(start_date)
    end_dt = iso_to_datetime(end_date)
    num_days = int((end_dt - start_dt).days)
    return [datetime_to_iso(
            start_dt + datetime.timedelta(i)) for i in range(num_days + 1)]


def next_date(date):
    """ Get date of day after input date in ISO 8601 format.

    For instance, if input date is ``'2017-03-12'``, the function returns
    ``'2017-03-13'``

    Args:
        date (str): Input date in ISO 8601 format.

    Returns:
        str: Date after input date in ISO 8601 format.

    """
    dtm = iso_to_datetime(date)
    return datetime_to_iso(dtm + datetime.timedelta(1))


def prev_date(date):
    """ Get date of day previous to input date in ISO 8601 format

    For instance, if input date is ``'2017-03-12'``, the function returns
    ``'2017-03-11'``

    Args:
        date (str): Input date in ISO 8601 format.

    Returns:
        str: Date before input date in ISO 8601 format.

    """
    dtm = iso_to_datetime(date)
    return datetime_to_iso(dtm - datetime.timedelta(1))


def iso_to_datetime(date):
    """ Convert ISO 8601 time format to datetime format

    This function converts a date in ISO format, e.g. ``2017-09-14`` to
    a ``datetime`` instance, e.g. ``datetime.datetime(2017,9,14,0,0)``

    Args:
        date (str): Input date in ISO 8601 format.

    Returns:
        datetime instance.

    """
    chunks = list(map(int, date.split('T')[0].split('-')))
    return datetime.datetime(chunks[0], chunks[1], chunks[2])


def datetime_to_iso(date, only_date=True):
    """ Convert datetime format to ISO 8601 time format.

    This function converts a date in datetime instance,
    e.g. ``datetime.datetime(2017,9,14,0,0)`` to ISO format,e.g. ``2017-09-14``

    Args:
        date (datetime instance): Datetime instance to convert.
        only_date (bool): Whether or not to return date only or also
                time information. Default is ``True``.

    Returns:
        str: Date in ISO 8601 format.

    """
    if only_date:
        return date.isoformat().split('T')[0]
    return date.isoformat()


def get_current_date():
    """ Get current date in ISO 8601 format.

    Returns:
        str: Current date in ISO 8601 format.

    """
    date = datetime.datetime.now()
    return datetime_to_iso(date)


def is_valid_time(time):
    """ Check if input string represents a valid time/date stamp.

    Args:
        time (str): A string containing a time/date stamp.

    Returns:
        Bool: ``True`` if string is a valid time/date stamp,
              ``False`` otherwise.

    """
    import dateutil.parser
    try:
        dateutil.parser.parse(time)
        return True
    except BaseException:
        return False


def parse_time(time_str):
    """ Parse input time/date string as ISO 8601 string.

    Args:
        time_str (str): Time/date string to parse.

    Returns:
        str: Parsed string in ISO 8601 format.

    """
    import dateutil.parser
    if len(time_str) < 8:
        raise ValueError('Invalid time string {}.\n'
                         'Please specify time in formats YYYY-MM-DD or '
                         'YYYY-MM-DDTHH:MM:SS'.format(time_str))
    time = dateutil.parser.parse(time_str)
    if len(time_str) <= 10:
        return time.date().isoformat()
    return time.isoformat()


def one_hour():
    """ Define one_hour for calculating.
    """
    return datetime.timedelta(hours=1)
