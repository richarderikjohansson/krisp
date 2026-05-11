# move later
import numpy as np
import pyarts
from pyarts.arts import GriddedField3

from krisp.filesystem.paths import find_arts_paths
from krisp.data.tools import interp_from_ecmwf_to_pret
from krisp.data.tools import make_ds_for_arts
from krisp.physics.atmosphere import altitude_from_pressure


class RTandAtmosphereRetrieval:
    def __init__(self, obj):
        self.data = obj.data
        self.arts = obj.arts
        self.attrs = obj.attrs
        self.config = obj.config

    def get_arts_paths(self):
        arts_paths = find_arts_paths(self.attrs)
        self.lines_path = arts_paths.lines
        self.cia_path = arts_paths.cia
        self.atm_base_path = arts_paths.atmosphere_base

    def set_absorption(self):
        fs = self.config.f_start
        fe = self.config.f_end
        abs_species = []
        for s in self.config.abs_species:
            if "O3" in s:
                fill = s + f"-*-{fs - 1e9}-{fe + 1e9}"
                abs_species.append(fill)
            elif "PWR" in s:
                fill = s + f"-{fs - 1e9}-{fe + 1e9}"
                abs_species.append(fill)
            else:
                abs_species.append(s)
        self.arts.abs_speciesSet(species=np.array(abs_species))
        self.arts.abs_lines_per_speciesReadSpeciesSplitCatalog(basename=self.lines_path)
        self.arts.abs_cia_dataReadSpeciesSplitCatalog(basename=self.cia_path)

    def set_radiative_transfer(self):
        self.arts.jacobianOff()
        self.arts.cloudboxOff()
        self.arts.stokes_dim = self.config.stokes_dim

        @pyarts.workspace.arts_agenda(ws=self.arts, set_agenda=True)
        def gas_scattering_agenda(ws):
            ws.Ignore(ws.rtp_vmr)
            ws.gas_scattering_coefAirSimple()
            ws.gas_scattering_matRayleigh()

        self.arts.iy_unit = self.config.iy_unit
        self.arts.ppath_lmax = self.config.ppath_lmax
        self.arts.propmat_clearsky_agendaAuto()

    def set_atmosphere(self):
        self.arts.AtmosphereSet1D()
        self.arts.PlanetSet(option="Earth")
        self.arts.nlteOff()
        self.arts.AtmRawRead(basename=self.atm_base_path)

        self.t_ret = interp_from_ecmwf_to_pret(data=self.data, product="temp")
        self.h2o_vmr = interp_from_ecmwf_to_pret(data=self.data, product="h2o")

        z_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.data.zret.values,
                self.data.pret.values,
                config=self.config,
            )
        )

        t_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.t_ret,
                self.data.pret.values,
                config=self.config,
            )
        )

        h2o_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.h2o_vmr,
                self.data.pret.values,
                config=self.config,
            )
        )

        o3_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.data.apriori.values,
                self.data.pret.values,
                config=self.config,
            )
        )

        self.arts.t_field_raw = t_gf
        self.arts.z_field_raw = z_gf
        self.arts.vmr_field_raw.value[0] = o3_gf
        self.arts.vmr_field_raw.value[1] = h2o_gf
        self.arts.lat_true = [self.config.lat]
        self.arts.lon_true = [self.config.lon]
        self.arts.z_surface = [[self.data.zret.values[0]]]
        self.arts.AtmFieldsCalc()

    def set_los(self):
        self.arts.sensor_pos = [[self.data.zret.values[0] + 20]]
        self.arts.sensor_los = [[self.attrs.za.values]]
        self.arts.AntennaOff()
        self.arts.ArrayOfTimeSetConstant(
            self.arts.sensor_time,
            1,
            pyarts.arts.Time(0),
        )

        @pyarts.workspace.arts_agenda(ws=self.arts)
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.FlagOn(ws.sensor_norm)
            ws.sensor_responseInit()
            ws.backend_channel_responseGaussianConstant(fwhm=self.config.f_res)
            ws.sensor_responseBackend()

        self.arts.Copy(self.arts.sensor_response_agenda, sensor_response_agenda)
        self.arts.AgendaExecute(self.arts.sensor_response_agenda)

    def execute(self):
        self.get_arts_paths()
        self.set_absorption()
        self.set_radiative_transfer()
        self.set_atmosphere()
        self.set_los()
