import numpy as np
from pysrc.preference.Enumclasses import FieldType
from pysrc.preference.Constants import GeoConstants
from pysrc.auxiliary.GeoMathKit import GeoMathKit


class ConvertSHC:
    def __init__(self, cqlm: np.ndarray, sqlm: np.ndarray, field_type: FieldType, ln):
        self.single_data_flag = False
        if cqlm.ndim == 2:
            self.single_data_flag = True
            cqlm, sqlm = GeoMathKit.getCSGridin3d(cqlm, sqlm)

        self.cqlm, self.sqlm = cqlm, sqlm

        self.nmax = np.shape(cqlm)[1] - 1

        self.field_type = field_type

        self.ln = ln

    def convert_to(self, field_type: FieldType):
        self._convertToDimensionless()
        self._convertFromDimensionless(field_type)

        if self.single_data_flag:
            self.cqlm, self.sqlm = self.cqlm[0], self.sqlm[0]

        return self

    def _convertToDimensionless(self):
        density_water = GeoConstants.density_water
        density_earth = GeoConstants.density_earth
        radius_e = GeoConstants.radius_earth

        if self.field_type is FieldType.Dimensionless:
            pass

        elif self.field_type is FieldType.EWH:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 * density_water / (
                    radius_e * density_earth)

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.sqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.field_type = FieldType.Dimensionless

        elif self.field_type is FieldType.Density:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 / (radius_e * density_earth)

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.cqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.field_type = FieldType.Dimensionless

        elif self.field_type is FieldType.Geoid:
            self.cqlm = self.cqlm / radius_e
            self.sqlm = self.sqlm / radius_e

            self.type = FieldType.Dimensionless

        else:
            return -1

        return self

    def _convertFromDimensionless(self, field_type: FieldType):
        density_water = GeoConstants.density_water
        density_earth = GeoConstants.density_earth
        radius_e = GeoConstants.radius_earth

        if field_type is FieldType.Dimensionless:
            pass

        elif field_type is FieldType.EWH:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / (
                    3 * density_water)

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.sqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.type = FieldType.EWH

        elif field_type is FieldType.Density:
            ln = self.ln[:self.nmax + 1]
            kn = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / 3

            self.cqlm = np.einsum('l,qlm->qlm', kn, self.cqlm)
            self.sqlm = np.einsum('l,qlm->qlm', kn, self.sqlm)

            self.type = FieldType.Density

        elif field_type is FieldType.Geoid:
            self.cqlm = radius_e * self.cqlm
            self.sqlm = radius_e * self.sqlm

            self.type = FieldType.Geoid

        else:
            return -1

        return self

    def getCS(self):
        return self.cqlm, self.sqlm
