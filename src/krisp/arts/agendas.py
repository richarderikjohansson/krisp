import pyarts
from pyarts.workspace import arts_agenda


def default_agendas(ws: pyarts.Workspace) -> pyarts.Workspace:
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.ppath_agendaSet(option="FollowSensorLosPath")  # Line of sight
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.water_p_eq_agendaSet(option="MK05")

    @arts_agenda(ws=ws, set_agenda=True)
    def inversion_iterate_agenda(ws):
        """Custom inversion iterate agenda to ignore bad partition functions."""
        ws.Ignore(ws.inversion_iteration_counter)

        ws.xClip(ijq=0, limit_low=0.00000000001, limit_high=0.00002)

        ws.x2artsAtmAndSurf()
        ws.x2artsSensor()
        ws.atmfields_checkedCalc(negative_vmr_ok=True)
        ws.atmgeom_checkedCalc()
        ws.yCalc()  # (y=ws.yf)
        ws.VectorAddElementwise(ws.yf, ws.y, ws.y_baseline)
        ws.jacobianAdjustAndTransform()

    return ws
