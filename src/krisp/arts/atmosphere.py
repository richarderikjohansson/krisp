# move later
import numpy as np
import pyarts
from pyarts.arts import GriddedField3

from krisp.filesystem.paths import find_arts_paths
from krisp.data.tools import interp_from_ecmwf_to_pret
from krisp.data.tools import make_ds_for_arts
from krisp.physics.atmosphere import altitude_from_pressure


class AtmosphereAndRT:
    def __init__(self, obj):
        self.retobj = obj
        self.retconf = obj.config

    def get_arts_paths(self):
        arts_paths = find_arts_paths(self.retobj.attrs)
        self.lines_path = arts_paths.lines
        self.cia_path = arts_paths.cia
        self.atm_base_path = arts_paths.atmosphere_base

    def set_absorption(self):
        fs = self.retconf.f_start
        fe = self.retconf.f_end
        abs_species = []
        for s in self.retconf.abs_species:
            if "O3" in s:
                fill = s + f"-*-{fs - 1e9}-{fe + 1e9}"
                abs_species.append(fill)
            elif "PWR" in s:
                fill = s + f"-{fs - 1e9}-{fe + 1e9}"
                abs_species.append(fill)
            else:
                abs_species.append(s)
        self.retobj.arts.abs_speciesSet(species=np.array(abs_species))
        self.retobj.arts.abs_lines_per_speciesReadSpeciesSplitCatalog(basename=self.lines_path)
        self.retobj.arts.abs_cia_dataReadSpeciesSplitCatalog(basename=self.cia_path)

    def set_radiative_transfer(self):
        self.retobj.arts.jacobianOff()
        self.retobj.arts.cloudboxOff()
        self.retobj.arts.stokes_dim = self.retconf.stokes_dim

        @pyarts.workspace.arts_agenda(ws=self.retobj.arts, set_agenda=True)
        def gas_scattering_agenda(ws):
            ws.Ignore(ws.rtp_vmr)
            ws.gas_scattering_coefAirSimple()
            ws.gas_scattering_matRayleigh()

        self.retobj.arts.iy_unit = self.retconf.iy_unit
        self.retobj.arts.ppath_lmax = self.retconf.ppath_lmax
        self.retobj.arts.propmat_clearsky_agendaAuto()

    def set_atmosphere(self):
        self.retobj.arts.AtmosphereSet1D()
        self.retobj.arts.PlanetSet(option="Earth")
        self.retobj.arts.nlteOff()
        self.retobj.arts.AtmRawRead(basename=self.atm_base_path)

        z = altitude_from_pressure(self.retobj.data.p.values, self.retobj.data.temperature.values)
        self.z_ret = interp_from_ecmwf_to_pret(data=self.retobj.data, product=z)
        self.t_ret = interp_from_ecmwf_to_pret(data=self.retobj.data, product="temperature")
        self.h2o_vmr = interp_from_ecmwf_to_pret(data=self.retobj.data, product="h2o")
        #        apriori = interp_from_ecmwf_to_pret(
        #            data=self.retobj.data,
        #            product=self.retobj.data.o3.values,
        #        )

        z_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.z_ret,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        t_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.t_ret,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        h2o_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.h2o_vmr,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        o3_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.retobj.data.apriori.values,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        self.retobj.arts.t_field_raw = t_gf
        self.retobj.arts.z_field_raw = z_gf
        self.retobj.arts.vmr_field_raw.value[0] = o3_gf
        self.retobj.arts.vmr_field_raw.value[1] = h2o_gf
        self.retobj.arts.lat_true = [self.retconf.lat]
        self.retobj.arts.lon_true = [self.retconf.lon]
        self.retobj.arts.z_surface = [[self.z_ret[0]]]
        self.retobj.arts.AtmFieldsCalc()

    def set_los(self):
        self.retobj.arts.sensor_pos = [[self.z_ret[0] + 20]]
        self.retobj.arts.sensor_los = [[self.retobj.attrs.zenith]]
        self.retobj.arts.AntennaOff()

        @pyarts.workspace.arts_agenda(ws=self.retobj.arts)
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.FlagOn(ws.sensor_norm)
            ws.sensor_responseInit()
            ws.backend_channel_responseGaussianConstant(fwhm=self.retconf.f_res)
            ws.sensor_responseBackend()

        self.retobj.arts.Copy(self.retobj.arts.sensor_response_agenda, sensor_response_agenda)
        self.retobj.arts.AgendaExecute(self.retobj.arts.sensor_response_agenda)

    def execute(self):
        self.get_arts_paths()
        self.set_absorption()
        self.set_radiative_transfer()
        self.set_atmosphere()
        self.set_los()
