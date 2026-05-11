import pyarts
from pyarts.workspace import arts_agenda


def default_agendas(ws: pyarts.Workspace) -> pyarts.Workspace:
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.water_p_eq_agendaSet(option="MK05")

    return ws
