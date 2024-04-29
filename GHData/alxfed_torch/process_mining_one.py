# -*- coding: utf-8 -*-
"""pm4py
"""
from pm4py.algo.discovery.alpha import versions
from pm4py.objects.conversion.log import factory as log_conversion


ALPHA_VERSION_CLASSIC = 'classic'
ALPHA_VERSION_PLUS = 'plus'
VERSIONS = {ALPHA_VERSION_CLASSIC: versions.classic.apply,
            ALPHA_VERSION_PLUS: versions.plus.apply}


def apply(log, parameters=None, variant=ALPHA_VERSION_CLASSIC):
    return VERSIONS[variant](log_conversion.apply(log, parameters, log_conversion.TO_EVENT_LOG), parameters)


if __name__ == '__main__':
    print('\ndone')