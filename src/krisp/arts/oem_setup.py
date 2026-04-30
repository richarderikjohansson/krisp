import numpy as np
import pyarts


class RetrievalOEMInit:
    """
    TODO: This is very much not a finished product. Have to oversee all retrieval species and such
    """

    def __init__(self, obj):
        self.retobj = obj
        self.retconf = obj.config
        self.p_ret = obj.data.p_ret.values
        self.lat = obj.arts.lat_grid
        self.lon = obj.arts.lon_grid

    def set_ret_quantities(self):
        self.retobj.arts.retrievalDefInit()

        self.add_species(species="O3")
        self.add_polyfit()
        self.add_frequency_shift()
        self.add_Se()

        self.retobj.arts.retrievalDefClose()

    def add_species(self, species):
        # will change
        match species:
            case "O3":
                vec = np.full_like(self.retobj.data.p_ret.values, 1e-6)
                covmat = np.diag(vec)
                spec = str(self.retobj.arts.abs_species.value[0])
        self.retobj.arts.retrievalAddAbsSpecies(
            g1=self.p_ret,
            g2=np.array([0]),
            g3=np.array([0]),
            species=spec,
            unit="vmr",
        )
        self.retobj.arts.covmat_sxAddBlock(block=covmat)

    def add_polyfit(self):
        self.retobj.arts.retrievalAddPolyfit(poly_order=self.retconf.poly_order)

        for cov in self.retconf.poly_covs:
            self.retobj.arts.covmat_sxAddBlock(block=cov)

    def add_frequency_shift(self):
        self.retobj.arts.retrievalAddFreqShift(df=self.retconf.fshift_df)
        self.retobj.arts.covmat_sxAddBlock(block=self.retconf.fshift_cov)

    def add_Se(self):
        f_clip = self.retconf.f_clip
        vec = np.full_like(self.retobj.data.fb.values[f_clip:-f_clip], 0.7)
        sparse_block = pyarts.arts.Sparse()
        self.retobj.arts.DiagonalMatrix(sparse_block, vec)
        self.retobj.arts.covmat_seAddBlock(block=sparse_block)

    def define_outputs(self):
        self.retobj.arts.x = np.array([])
        self.retobj.arts.yf = np.array([])
        self.retobj.arts.jacobian = np.array([[]])
        self.retobj.arts.xaStandard()
