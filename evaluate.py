#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math

def RMSE(predict, correct):
    return math.sqrt(np.average((predict - correct) ** 2))
