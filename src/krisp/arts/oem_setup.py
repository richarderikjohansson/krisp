import numpy as np
import pyarts


class RetrievalOEMInit:
    """
    TODO: This is very much not a finished product. Have to oversee all retrieval species and such
    """

    def __init__(self, obj):
        self.data = obj.data
        self.arts = obj.arts
        self.attrs = obj.attrs
        self.config = obj.config

    def set_ret_quantities(self):
        self.arts.retrievalDefInit()

        self.add_species(species="O3")
        self.add_species(species="H2O")
        self.add_species(species="O2")
        self.add_polyfit()
        self.add_frequency_shift()
        self.add_Se()

        self.arts.retrievalDefClose()

    def add_species(self, species):
        # will change
        match species:
            case "O3":
                vec = np.full_like(self.data.pret.values, 0.5)
                covmat = np.diag(vec)
                spec = str(self.arts.abs_species.value[0])
            case "H2O":
                vec = np.full_like(self.data.pret.values, 0.5)
                covmat = np.diag(vec)
                spec = str(self.arts.abs_species.value[1])

            case "O2":
                vec = np.full_like(self.data.pret.values, 0.5)
                covmat = np.diag(vec)
                spec = str(self.arts.abs_species.value[2])

        self.arts.retrievalAddAbsSpecies(
            g1=self.data.pret.values,
            g2=np.array([0]),
            g3=np.array([0]),
            species=spec,
        )
        self.arts.covmat_sxAddBlock(block=covmat)

    def add_polyfit(self):
        self.arts.retrievalAddPolyfit(poly_order=self.config.poly_order)

        for cov in self.config.poly_covs:
            self.arts.covmat_sxAddBlock(block=cov)

    def add_frequency_shift(self):
        self.arts.retrievalAddFreqShift(df=self.config.fshift_df)
        self.arts.covmat_sxAddBlock(block=self.config.fshift_cov)

    def add_Se(self):
        f_clip = self.config.f_clip
        vec = np.full_like(self.data.fb.values[f_clip:-f_clip], 0.7)
        sparse_block = pyarts.arts.Sparse()
        self.arts.DiagonalMatrix(sparse_block, vec)
        self.arts.covmat_seAddBlock(block=sparse_block)

    def define_outputs(self):
        self.arts.x = np.array([])
        self.arts.yf = np.array([])
        self.arts.jacobian = np.array([[]])
        self.arts.xaStandard()
