import os
import win32com.client as win32
import numpy as np


class AspenPlusModel:
    def __init__(self, bkp_folder, comp_names=None):
        self.bkp_folder = bkp_folder
        self.comp_names = comp_names or ['H2', 'CO', 'CO2', 'H2O', 'MEOH', 'ETOH', 'DME']
        self.simulation = None
        self.cache = {}

    def initialize_aspen_plus(self):
        """Initialize the Aspen Plus simulation."""
        try:
            print("Initializing Aspen Plus simulation...")
            path_surr = self.find_bkp_file()
            if not path_surr:
                raise FileNotFoundError("No .bkp file found in the specified directory.")

            # Load Aspen Plus simulation
            self.simulation = win32.Dispatch('Apwn.Document')
            self.simulation.InitFromArchive2(os.path.abspath(path_surr))
            self.simulation.Visible = False
            self.simulation.SuppressDialogs = True  # Suppress pop-up dialogs
            self.simulation.Reinit()
            print("Aspen Plus simulation initialized successfully.")

        except Exception as e:
            print(f"Error during Aspen Plus initialization: {e}")
            raise e

    def find_bkp_file(self):
        """Find the Aspen Plus .bkp file in the specified folder."""
        for file in os.listdir(self.bkp_folder):
            if file.endswith('.bkp'):
                return os.path.join(self.bkp_folder, file)
        return None

    def _cache_key(self, xk):
        """Generate a cache key from the input array."""
        return tuple(round(x, 2) for x in xk)

    def run_aspen_plus(self, xk):
        """Run the Aspen Plus simulation with the provided inputs."""
        if self.simulation is None:
            raise RuntimeError("Aspen Plus simulation is not initialized. Call 'initialize_simulation()' first.")

        cache_key = self._cache_key(xk)
        if cache_key in self.cache:
            # print("Cache hit: Returning cached result.")
            return self.cache[cache_key], True

        dw = None  # Initialize dw to ensure it's always defined
        try:
            # Set input variables
            vas_T = r"\Data\Streams\S24\Input\TEMP\MIXED"
            vas_P = r"\Data\Streams\S24\Input\PRES\MIXED"
            vas_comp = r"\Data\Streams\S24\Input\FLOW\MIXED"

            for ist in range(len(xk)):
                if ist == 0:
                    vst = vas_T
                elif ist == 1:
                    vst = vas_P
                else:
                    vst = vas_comp + "\\" + self.comp_names[ist - 2]

                node = self.simulation.Tree.FindNode(vst)
                if node is None:
                    raise ValueError(f"Node not found for variable index {ist} ({vst})")
                node.Value = xk[ist]

            # Add inert gases
            inert_gases = {'N2': 298, 'CH4': 738}
            for gas, value in inert_gases.items():
                node = self.simulation.Tree.FindNode(rf"\Data\Streams\S24\Input\FLOW\MIXED\{gas}")
                if node is None:
                    raise ValueError(f"Node for {gas} not found.")
                node.Value = value

            # Run the Aspen Plus simulation
            print("Running Aspen Plus simulation...")
            self.simulation.Engine.Run2()

            # Retrieve output variables
            dw_T = r"\Data\Streams\X-R201D\Output\TEMP_OUT\MIXED"
            dw_P = r"\Data\Streams\X-R201D\Output\PRES_OUT\MIXED"
            dw_comp = r"\Data\Streams\X-R201D\Output\MASSFLOW\MIXED"

            dw = np.zeros(len(xk))
            for ist in range(len(xk)):
                if ist == 0:
                    vst = dw_T
                elif ist == 1:
                    vst = dw_P
                else:
                    vst = dw_comp + "\\" + self.comp_names[ist - 2]

                node = self.simulation.Tree.FindNode(vst)
                if node is None:
                    raise ValueError(f"Node not found for output variable index {ist} ({vst})")
                dw[ist] = node.Value

            if dw is None:
                print(f"Aspen Plus Simulation Failed. No feasibility.")
                return np.zeros(len(xk)), False
            # Cache the result
            self.cache[cache_key] = dw
            return dw, True

        except Exception as e:
            print(f"Error during Aspen Plus simulation run: {e}")
            return np.zeros(len(xk)), False
