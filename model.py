import pyomo.environ as pyo


def apply_constraints(m):
    # ==========================================
    # Cost Distributions & Objective
    # ==========================================

    @m.Constraint(m.region, m.sector, m.y)
    def calc_cost_capture(m, i, s, t):
        return m.cost_capture[i, s, t] == sum(
            # OPEX uses vintage 'v' price (uniform across regions: s, k, v)
            sum(
                m.c_cap_opex[s, k, v] * m.q_CO2_cap[i, s, k, v, t]
                for v in m.v
                if v <= t
            )
            # CAPEX is paid once for new capacity built in year 't' (uniform across regions: s, k, t)
            + m.c_cap_capex[s, k, t] * m.Q_new[i, s, k, t]
            for k in m.tech
        )

    @m.Constraint(m.region, m.sector, m.y)
    def calc_cost_ets(m, i, s, t):
        return m.cost_ets[i, s, t] == m.c_ETS[t] * m.q_CO2_ETS[i, s, t]

    @m.Constraint(m.region, m.y)
    def calc_cost_transport(m, i, t):
        return m.cost_transport[i, t] == sum(
            m.c_trans[i, j] * m.q_CO2_trans[i, j, t] for j in m.region if i != j
        )

    @m.Constraint(m.region, m.y)
    def calc_cost_sink(m, i, t):
        return m.cost_sink[i, t] == sum(
            # Storage and injection costs are uniform across regions (b, t)
            m.c_sink_capex[b, t] * m.sink_bdv[i, b, t]
            + m.c_inj_opex[b, t] * m.q_CO2_inj[i, b, t]
            for b in m.sink_type
        )

    @m.Constraint()
    def calc_total_costs(m):
        total_capture_ets = sum(
            m.cost_capture[i, s, t] + m.cost_ets[i, s, t]
            for i in m.region
            for s in m.sector
            for t in m.y
        )
        total_transport_sink = sum(
            m.cost_transport[i, t] + m.cost_sink[i, t] for i in m.region for t in m.y
        )
        return m.costs == total_capture_ets + total_transport_sink

    # Objective minimizes total aggregated costs
    m.Obj = pyo.Objective(expr=m.costs, sense=pyo.minimize)

    # ==========================================
    # Constraints
    # ==========================================

    # Eq 1: CSU Obligation
    @m.Constraint(m.region, m.y)
    def csu_obligation(m, i, t):
        ng_sum = sum(
            m.f_NG_intern[i, j, s, t] for j in m.region for s in m.sector
        ) + sum(m.f_NG_import[i, s, t] for s in m.sector)
        return m.g[t] * ng_sum * m.psi[t] <= m.CSU_use[i, t]

    # Eq 2: CSU Balance (Active years only via m.y_tilde)
    @m.Constraint(m.region, m.y_tilde)
    def csu_balance(m, i, t):
        return (
            m.CSU_balance[i, t]
            == m.CSU_balance[i, t - 1]
            + m.CSU_generate[i, t]
            + m.CSU_buy[i, t]
            - m.CSU_sell[i, t]
            - m.CSU_use[i, t]
        )

    # Eq 3: CSU Market Clearing
    @m.Constraint(m.y)
    def csu_market_clearing(m, t):
        return sum(m.CSU_buy[i, t] for i in m.region) == sum(
            m.CSU_sell[i, t] for i in m.region
        )

    # Eq 4: CSU Generation
    @m.Constraint(m.region, m.y)
    def csu_generation(m, i, t):
        return m.CSU_generate[i, t] == sum(m.q_CO2_inj[i, b, t] for b in m.sink_type)

    # Eq 5: CO2 Network Balance (Vintage-aware)
    @m.Constraint(m.region, m.y)
    def co2_network_balance(m, i, t):
        captured = sum(
            m.q_CO2_cap[i, s, k, v, t]
            for s in m.sector
            for k in m.tech
            for v in m.v
            if v <= t
        )
        imported = sum(m.q_CO2_trans[j, i, t] for j in m.region if j != i)
        exported = sum(m.q_CO2_trans[i, j, t] for j in m.region if j != i)
        injected = sum((1 / m.eta) * m.q_CO2_inj[i, b, t] for b in m.sink_type)
        return captured + imported == exported + injected

    # Eq 6: ETS Emitter (Vintage-aware)
    @m.Constraint(m.region, m.sector, m.y)
    def ets_emitter(m, i, s, t):
        emissions = sum(m.f_NG_intern[j, i, s, t] for j in m.region) * m.psi[t]
        captured = sum(
            m.q_CO2_cap[i, s, k, v, t] for k in m.tech_point for v in m.v if v <= t
        )
        return m.q_CO2_ETS[i, s, t] == emissions - captured

    # Eq 7: Capture Limit (Vintage-aware)
    @m.Constraint(m.region, m.sector, m.tech_point, m.y)
    def capture_limit(m, i, s, k, t):
        emissions = sum(m.f_NG_intern[j, i, s, t] for j in m.region) * m.psi[t]
        return (
            sum(m.q_CO2_cap[i, s, k, v, t] for v in m.v if v <= t)
            <= m.mu[s, k] * emissions
        )

    # Eq 8 & 9: Capacity Expansion and Init
    @m.Constraint(m.region, m.sector, m.tech, m.y)
    def cap_expansion(m, i, s, k, t):
        if t == m.y.first():
            return m.Q_cap[i, s, k, t] == m.Q_init[i, s, k] + m.Q_new[i, s, k, t]
        return m.Q_cap[i, s, k, t] == m.Q_cap[i, s, k, t - 1] + m.Q_new[i, s, k, t]

    # Eq 10: Vintage Capacity Upper Bound
    @m.Constraint(m.region, m.sector, m.tech, m.v, m.y)
    def cap_max_vintage(m, i, s, k, v, t):
        if v > t:
            return m.q_CO2_cap[i, s, k, v, t] == 0

        if v == m.y.first():
            return m.q_CO2_cap[i, s, k, v, t] <= m.Q_init[i, s, k] + m.Q_new[i, s, k, v]
        else:
            return m.q_CO2_cap[i, s, k, v, t] <= m.Q_new[i, s, k, v]

    # Eq 11 & 12: Capacity Expansion Limit (Active years via y_tilde)
    @m.Constraint(m.region, m.sector, m.tech, m.y_tilde)
    def cap_exp_limit(m, i, s, k, t):
        return m.Q_new[i, s, k, t] - m.Q_new[i, s, k, t - 1] <= m.Q_delta_abs[
            i, s, k
        ] + (m.Q_IR_cap[i, s, k] * m.Q_new[i, s, k, t - 1])

    # Eq 13: Deployment Limit (Point-source sector bounds)
    @m.Constraint(m.region, m.sector, m.tech_point, m.y)
    def deployment_limit(m, i, s, k, t):
        emissions = sum(m.f_NG_intern[j, i, s, t] for j in m.region) * m.psi[t]
        return m.Q_cap[i, s, k, t] <= m.rho[s, t] * emissions

    # Eq 14: Sink Explore
    @m.Constraint(m.region, m.sink_type)
    def sink_explore(m, i, b):
        return sum(m.sink_bdv[i, b, t] for t in m.y) <= 1

    # Eq 15: Sink Active Capacity
    @m.Constraint(m.region, m.sink_type, m.y)
    def sink_active_cap(m, i, b, t):
        valid_years = [tau for tau in m.y if tau <= t - m.sink_timedelay[i, b]]
        if not valid_years:
            return m.sink_active_cap[i, b, t] == 0
        return (
            m.sink_active_cap[i, b, t]
            == sum(m.sink_bdv[i, b, tau] for tau in valid_years)
            * m.sink_injection_rate[i, b]
        )

    # Eq 16: Sink Injection Limit
    @m.Constraint(m.region, m.sink_type, m.y)
    def sink_inj_limit(m, i, b, t):
        return m.q_CO2_inj[i, b, t] <= m.sink_active_cap[i, b, t]

    # Eq 17: Sink Cumulative Cap
    @m.Constraint(m.region, m.sink_type, m.y)
    def sink_cum_cap(m, i, b, t):
        valid_years = [tau for tau in m.y if tau <= t]
        return (
            sum(m.q_CO2_inj[i, b, tau] for tau in valid_years) <= m.sink_block_cap[i, b]
        )

    return m