# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2019 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Helpers for author disambiguation.
... author tay

"""


def get_authors(s):
    """Get author full name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized author name
    """
    res = ' '
    v = s["authors"] if "authors" in s and s["authors"] is not None and len(s['authors']) else " "
    if v == " ":
        return res
    name_list = []
    if v:
        for item in v:
            name_list.append(item['name'])
        res = " ".join(name_list) if len(name_list) != 0 else " "
    return res


def get_author_affiliations(s):
    """Get author affiliation from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized affiliation name
    """
    res = ' '
    v = s["authors"] if "authors" in s and s["authors"] is not None and len(s['authors']) else " "
    if v == " ":
        return res
    aff_list = []
    if v:
        for item in v:
            if 'org' in item:
                aff_list.append(item['org'])
        res = " ".join(aff_list) if len(aff_list) != 0 else " "
    return res


def get_title(s):
    """Get publication's title from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Title of the publication
    """
    v = s["title"] if "title" in s and s['title'] is not None else " "
    if not v:
        v = " "
    return v


def get_venue(s):
    """Get journal's name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Journal's name
    """
    v = s["venue"] if 'venue' in s and s['venue'] is not None else ' '
    if not v:
        v = " "
    return v


def get_abstract(s):
    """Get author full name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized author name
    """
    v = s["abstract"] if 'abstract' in s and s['abstract'] is not None else ' '
    if not v:
        v = " "
    return v


def get_keywords(s):
    """Get keywords from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Keywords separated by a space
    """
    res = " "
    v = s["keywords"] if 'keywords' in s and s['keywords'] is not None and len(s['keywords']) else ' '
    if v[0] == '':
        return res
    if len(v):
        res = " ".join(v)
    else:
        res = " "
    return res


def get_year(s):
    """Get year from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: int
        Year of publication if present on the signature, -1 otherwise
    """
    v = s["year"] if 'year' in s and s['year'] is not None and s['year'] else -1
    return v