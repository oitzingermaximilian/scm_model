import pyomo.environ as pyo


def build_base_model(data_dict):
    """
    Initializes the Pyomo model with sets, parameters, and variables.
    Assumes data_dict contains the populated data from the Excel sheets.
    """
    m = pyo.ConcreteModel()

    # ==========================================
    # 1. SETS & INDICES
    # ==========================================
    m.y = pyo.Set(initialize=data_dict["y"], doc="Full modeling horizon (years)")
    m.y_tilde = pyo.Set(
        initialize=m.y, filter=lambda m, t: t != m.y.first(), doc="Active periods"
    )
    m.v = pyo.Set(initialize=data_dict["y"], doc="Vintage years (construction year)")

    m.region = pyo.Set(
        initialize=data_dict["region"], doc="Regions in the carbon management club"
    )
    m.sector = pyo.Set(
        initialize=data_dict["sector"], doc="Sectors driving natural gas demand"
    )

    # Technology Sets
    m.tech = pyo.Set(
        initialize=data_dict["tech"], doc="Available carbon capture technologies"
    )
    m.tech_point = pyo.Set(
        within=m.tech,
        initialize=data_dict["tech_point"],
        doc="Point-source technologies",
    )
    m.tech_rem = pyo.Set(
        within=m.tech,
        initialize=data_dict["tech_rem"],
        doc="Standalone removal technologies",
    )

    m.storage_types = pyo.Set(
        initialize=data_dict["storage_types"], doc="Geological sink and storage types"
    )

    m.sink_blocks = pyo.Set(
        initialize=data_dict["sink_blocks"], doc="Project-specific geological sink blocks"
    )

    # ==========================================
    # 2. PARAMETERS
    # ==========================================
    m.g = pyo.Param(
        m.y,
        initialize=data_dict["g"],
        default=0,
        doc="Geologically stored fraction (%)",
    )
    m.psi = pyo.Param(
        m.y,
        initialize=data_dict["psi"],
        default=0,
        doc="Carbon intensity factor of natural gas",
    )
    m.eta = pyo.Param(
        initialize=0.9,
        doc="Universal transmission/injection loss efficiency factor"
    )

    # Baseline Energy Flows
    m.f_NG_intern = pyo.Param(
        m.region, m.region, m.sector, m.y,
        initialize=data_dict["f_NG_intern"],
        default=0.0,
        doc="Bilateral natural gas flows by sector"
    )
    m.f_NG_import = pyo.Param(
        m.region, m.sector, m.y,
        initialize=data_dict["f_NG_import"],
        default=0.0,
        doc="Domestic NG usage or external imports by sector"
    )

    m.mu = pyo.Param(
        m.sector,
        m.tech,
        initialize=data_dict["mu"],
        default=0.0,
        doc="Max technological capture rate",
    )
    m.rho = pyo.Param(
        m.sector,
        m.y,
        initialize=data_dict["rho"],
        doc="Max deployment penetration limit per sector/year",
    )

    # Sink Parameters
    m.sink_timedelay = pyo.Param(
        m.region,
        m.sink_blocks,
        initialize=data_dict["sink_timedelay"],
        default=0,
        doc="Time delay for sink development",
    )
    m.sink_injection_rate = pyo.Param(
        m.region,
        m.sink_blocks,
        initialize=data_dict["sink_injection_rate"],
        default=0.0,
        doc="Max annual injection rate (tonnes)",
    )
    m.sink_block_cap = pyo.Param(
        m.region,
        m.sink_blocks,
        initialize=data_dict["sink_block_cap"],
        default=0.0,
        doc="Absolute cumulative storage capacity (tonnes)",
    )
    m.c_sink_capex = pyo.Param(
        m.region,
        m.sink_blocks,
        m.y,
        initialize=data_dict["c_sink_capex"],
        default=0.0,
        doc="Storage total capital cost (€)",
    )
    m.c_inj_opex = pyo.Param(
        m.region,
        m.sink_blocks,
        m.y,
        initialize=data_dict["c_sink_opex"],
        default=0.0,
        doc="Distance-adjusted sink injection & transport OPEX (€/tCO2)",
    )

    # Costs & Economics
    m.c_ETS = pyo.Param(
        m.y, initialize=data_dict["c_ETS"], doc="Exogenous EU-ETS CO2 price"
    )
    m.c_cap_opex = pyo.Param(
        m.sector,
        m.tech,
        m.y,
        initialize=data_dict["c_cap_opex"],
        default=0.0,
        doc="Operational cost for capture",
    )
    m.c_cap_capex = pyo.Param(
        m.sector,
        m.tech,
        m.y,
        initialize=data_dict["c_cap_capex"],
        default=0.0,
        doc="Investment cost for capture capacity",
    )
    m.c_trans = pyo.Param(
        m.region,
        m.region,
        initialize=data_dict["c_trans"],
        default=0.0,
        doc="Inter-regional onshore transport cost (€/tCO2)"
    )

    # Expansion Limits
    m.Q_init = pyo.Param(
        m.region,
        m.sector,
        m.tech,
        initialize=data_dict.get("Q_init", {}),
        default=0.0,
        doc="Existing capacity active at the start of the horizon",
    )
    m.Q_delta_abs = pyo.Param(
        m.region,
        m.sector,
        m.tech,
        initialize=data_dict["Q_delta_abs"],
        default=0,
        doc="Max absolute annual increase in new capacity",
    )
    m.Q_IR_cap = pyo.Param(
        m.region,
        m.sector,
        m.tech,
        initialize=data_dict["Q_IR_cap"],
        default=0.35,
        doc="Max relative increase rate of new capacity",
    )

    # ==========================================
    # 3. VARIABLES
    # ==========================================
    m.costs = pyo.Var(domain=pyo.Reals, doc="Aggregated total system costs")

    m.cost_capture = pyo.Var(
        m.region, m.sector, m.y,
        domain=pyo.Reals,
        doc="Capture CAPEX/OPEX by region, sector, year",
    )
    m.cost_ets = pyo.Var(
        m.region, m.sector, m.y,
        domain=pyo.Reals,
        doc="EU-ETS penalty by region, sector, year",
    )
    m.cost_transport = pyo.Var(
        m.region, m.y,
        domain=pyo.Reals,
        doc="Transport costs originating from region i per year",
    )
    m.cost_sink = pyo.Var(
        m.region, m.y,
        domain=pyo.Reals,
        doc="Sink CAPEX/OPEX in region i per year"
    )

    # Physical CO2 Flows
    m.q_CO2_cap = pyo.Var(
        m.region, m.sector, m.tech, m.v, m.y, domain=pyo.NonNegativeReals
    )
    m.q_CO2_ETS = pyo.Var(
        m.region, m.sector, m.y,
        domain=pyo.NonNegativeReals,
        doc="Residual CO2 emissions for EU-ETS",
    )
    m.q_CO2_trans = pyo.Var(
        m.region, m.region, m.y,
        domain=pyo.NonNegativeReals,
        doc="Physical CO2 transported",
    )
    m.q_CO2_inj = pyo.Var(
        m.region, m.sink_blocks, m.y,
        domain=pyo.NonNegativeReals,
        doc="Physical CO2 injected",
    )

    # Carbon Storage Units (CSUs)
    m.CSU_balance = pyo.Var(m.region, m.y, domain=pyo.NonNegativeReals)
    m.CSU_generate = pyo.Var(m.region, m.y, domain=pyo.NonNegativeReals)
    m.CSU_buy = pyo.Var(m.region, m.y, domain=pyo.NonNegativeReals)
    m.CSU_sell = pyo.Var(m.region, m.y, domain=pyo.NonNegativeReals)
    m.CSU_use = pyo.Var(m.region, m.y, domain=pyo.NonNegativeReals)

    # Capacity and Sinks
    m.Q_cap = pyo.Var(
        m.region, m.sector, m.tech, m.y,
        domain=pyo.NonNegativeReals,
        doc="Installed, active capture capacity",
    )
    m.Q_new = pyo.Var(
        m.region, m.sector, m.tech, m.y,
        domain=pyo.NonNegativeReals,
        doc="Endogenous addition of new capture capacity",
    )

    m.sink_bdv = pyo.Var(
        m.region, m.sink_blocks, m.y,
        domain=pyo.Binary,
        doc="1 if development of sink starts in year t",
    )
    m.sink_active_cap = pyo.Var(
        m.region, m.sink_blocks, m.y,
        domain=pyo.NonNegativeReals,
        doc="Active annual injection capacity",
    )

    return m