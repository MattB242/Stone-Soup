import pandas as pd
import numpy as np
import os
import julia
from pathlib import path

from julia.api import Julia
jl = Julia(runtime = compiled_modules=False)

from julia import Main

from julia import Pkg
env_path = os.path.join(os.path.dirname(__file__), "..", "Julia_Calls")
Pkg.activate(env_path)
Pkg.instantiate()  # Ensure deps are downloaded

from julia import WidebandDoA
from julia import ReversibleJump
from julia import Distributions
from julia import JuliaBayes

from .base import Detector
from ..reader.file import FileReader

class CSVReader(FileReader):
    """Concrete file reader that loads CSVs into NumPy arrays."""

    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)

    def read(self):
        """Load the CSV file into a NumPy array."""
        return np.loadtxt(self.path, delimiter=",")
class Bayesian_DoA(Detector):
    """Returns a detection dictionary for a given timestamp"""
    def __init__(self, csv_path):
        super().__init__()
        self.sensor = CSVReader(path=csv_path)
    def process(self, n_sensor, dx,fs, secs, n_samples):
        arr = self.sensor.read()

        alpha, beta = 0,0
        source_prior = Distributions.InverseGamma(0.01, 0.01)
        order_prior = Distributions.truncated(Distributions.NegativeBinomial(1/2+0.1, 0.1/(0.1+1)), 0, n_sensor-1)
        model = WidebandDoA.WidebandIsoIsoModel(
            fs * secs, dx, fs, source_prior, alpha, beta, order_prior = order_prior
        )

        cond = WidebandDoA.WidebandConditioned(model, arr)
        prop = WidebandDoA.UniformNormalLocalProposal(0.0, 2.0)
        jump = WidebandDoA.IndepJumpProposal(prop)
        mcmc = WidebandDoA.SliceSteppingOut([2.0,2.0])

        rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump,mcmc,jump_rate = 0.9)
        n_samples = n_samples

        Main.eval("""
        using Random123
        seed = (0x97dc910eaebcfba, 0x741d36b68bef6415)
        rng = Random123.Philox4x(UInt64, seed, 8
        """)

        Main.eval("""
        function make_param(phi, loglambda)
            WidebandIsoIsoParam{Float64}(phi, loglambda)
        end
        """)

        param = [Main.make_param(0.0,0.0)]
        initial_order = 0

        samples, stats = ReversibleJump.sample(
            Main.rng,
            rjmcmc,
            cond,
            n_samples,
            initial_order,
            param,
            show_progress=False
        )


