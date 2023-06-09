from enum import Enum


class FieldType(Enum):
    Dimensionless = 0
    EWH = 1
    Geoid = 2
    Density = 3


class LoveNumberType(Enum):
    PREM = 1
    AOD04 = 2
    Wang = 3
    IERS = 4
