import pyomo.environ as pyo


def apply_constraints(m, discount_rate=0.05):
    # ==========================================
    # EXTRACT PARAMETERS TO BYPASS PYOMO BUGS
    # ==========================================
    dict_delay = m.sink_timedelay.extract_values()
    dict_inj_rate = m.sink_injection_rate.extract_values()
    dict_cap = m.sink_block_cap.extract_values()

    dict_sink_capex = m.c_sink_capex.extract_values()
    dict_inj_opex = m.c_inj_opex.extract_values()

    # NEW EXTRACTS FOR STAGED DEVELOPMENT
    dict_stage = m.sink_stage.extract_values()
    dict_parent = m.sink_parent.extract_values()

    base_year = m.y.first()

    # ==========================================
    # Cost Distributions & Objective (with Discounting)
    # ==========================================
    @m.Constraint(m.region, m.sector, m.y)
    def calc_cost_capture(m, i, s, t):
        return m.cost_capture[i, s, t] == sum(
            # 1. CAPEX: Tied to newly built capacity in year t
            m.c_cap_capex[s, k, t] * m.Q_new[i, s, k, t]

            # 2. Fixed OPEX: Tied to total installed capacity in year t
            + m.c_cap_fix_opex[s, k, t] * m.Q_cap[i, s, k, t]

            # 3. Variable OPEX: Tied to the actual volume of CO2 captured
            + sum(
                m.c_cap_var_opex[s, k, v] * m.q_CO2_cap[i, s, k, v, t]
                for v in m.v if v <= t
            )
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
            dict_sink_capex.get((i, b, t), 0.0) * m.sink_bdv[i, b, t]
            + dict_inj_opex.get((i, b, t), 0.0) * m.q_CO2_inj[i, b, t]
            for b in m.sink_blocks
        )

    @m.Constraint()
    def calc_total_costs(m):
        total_capture_ets_npv = sum(
            (m.cost_capture[i, s, t] + m.cost_ets[i, s, t]) / ((1 + discount_rate) ** (t - base_year))
            for i in m.region
            for s in m.sector
            for t in m.y
        )
        total_transport_sink_npv = sum(
            (m.cost_transport[i, t] + m.cost_sink[i, t]) / ((1 + discount_rate) ** (t - base_year))
            for i in m.region
            for t in m.y
        )
        # ==========================================
        # REGULARIZATION: Prevent "ghost" CSU trading
        # ==========================================
        epsilon = 1e-3
        csu_trading_penalty = sum(
            epsilon * m.CSU_sell[i, t] / ((1 + discount_rate) ** (t - base_year))
            for i in m.region
            for t in m.y
        )

        return m.costs == total_capture_ets_npv + total_transport_sink_npv + csu_trading_penalty

    m.Obj = pyo.Objective(expr=m.costs, sense=pyo.minimize)

    # ==========================================
    # CSU OBLIGATION & SPLIT BALANCE
    # ==========================================
    @m.Constraint(m.region, m.y)
    def csu_obligation(m, i, t):
        ng_sum = sum(
            m.f_NG_intern[i, j, s, t] for j in m.region for s in m.sector
        ) + sum(m.f_NG_import[i, s, t] for s in m.sector)
        return m.g[t] * ng_sum * m.psi[t] == m.CSU_use[i, t]

    @m.Constraint(m.region, m.y)
    def csu_total_use(m, i, t):
        return m.CSU_use[i, t] == m.CSU_use_local[i, t] + m.CSU_use_bought[i, t]

    @m.Constraint(m.region, m.y)
    def csu_local_balance(m, i, t):
        if t == base_year:
            return m.CSU_banked_local[i, t] == m.CSU_generate[i, t] - m.CSU_use_local[i, t] - m.CSU_sell[i, t] - m.CSU_cancel[i, t]
        else:
            return m.CSU_banked_local[i, t] == m.CSU_banked_local[i, t - 1] + m.CSU_generate[i, t] - m.CSU_use_local[
                i, t] - m.CSU_sell[i, t] - m.CSU_cancel[i, t]

    @m.Constraint(m.region, m.y)
    def csu_bought_balance(m, i, t):
        if t == base_year:
            return m.CSU_banked_bought[i, t] == m.CSU_buy[i, t] - m.CSU_use_bought[i, t] - m.CSU_cancel_bought[i, t]
        else:
            return m.CSU_banked_bought[i, t] == m.CSU_banked_bought[i, t - 1] + m.CSU_buy[i, t] - m.CSU_use_bought[i, t] - m.CSU_cancel_bought[i, t]

    @m.Constraint(m.y)
    def csu_market_clearing(m, t):
        return sum(m.CSU_buy[i, t] for i in m.region) == sum(m.CSU_sell[i, t] for i in m.region)

    @m.Constraint(m.region, m.y)
    def csu_generation(m, i, t):
        return m.CSU_generate[i, t] == sum(m.q_CO2_inj[i, b, t] for b in m.sink_blocks)

    # ==========================================
    # NO BANKING ALLOWED FROM 2050 ONWARDS
    # ==========================================
    @m.Constraint(m.region, m.y)
    def csu_no_banking_2050(m, i, t):
        if t >= 2050:
            return m.CSU_banked_local[i, t] + m.CSU_banked_bought[i, t] == 0
        return pyo.Constraint.Skip

    # ==========================================
    # CO2 NETWORK & CAPTURE
    # ==========================================
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

        injected = sum((1 / m.eta) * m.q_CO2_inj[i, b, t] for b in m.sink_blocks)

        return captured + imported == exported + injected

    @m.Constraint(m.region, m.sector, m.y)
    def ets_emitter(m, i, s, t):
        emissions = (
                            sum(m.f_NG_intern[j, i, s, t] for j in m.region if j != i)
                            + m.f_NG_import[i, s, t]
                    ) * m.psi[t]

        captured = sum(
            m.q_CO2_cap[i, s, k, v, t] for k in m.tech_point for v in m.v if v <= t
        )
        return m.q_CO2_ETS[i, s, t] == emissions - captured

    @m.Constraint(m.region, m.sector, m.tech_point, m.y)
    def capture_limit(m, i, s, k, t):
        emissions = (
                            sum(m.f_NG_intern[j, i, s, t] for j in m.region if j != i)
                            + m.f_NG_import[i, s, t]
                    ) * m.psi[t]

        return (
                sum(m.q_CO2_cap[i, s, k, v, t] for v in m.v if v <= t)
                <= m.mu[s, k] * emissions
        )

    # ==========================================
    # CAPACITY & EXPANSION
    # ==========================================
    @m.Constraint(m.region, m.sector, m.tech, m.y)
    def cap_expansion(m, i, s, k, t):
        if t == base_year:
            return m.Q_cap[i, s, k, t] == m.Q_init[i, s, k] + m.Q_new[i, s, k, t]
        return m.Q_cap[i, s, k, t] == m.Q_cap[i, s, k, t - 1] + m.Q_new[i, s, k, t]

    @m.Constraint(m.region, m.sector, m.tech, m.v, m.y)
    def cap_max_vintage(m, i, s, k, v, t):
        if v > t:
            return m.q_CO2_cap[i, s, k, v, t] == 0

        if v == base_year:
            return m.q_CO2_cap[i, s, k, v, t] <= m.Q_init[i, s, k] + m.Q_new[i, s, k, v]
        else:
            return m.q_CO2_cap[i, s, k, v, t] <= m.Q_new[i, s, k, v]

    @m.Constraint(m.region, m.sector, m.tech, m.y_tilde)
    def cap_exp_limit(m, i, s, k, t):
        return m.Q_new[i, s, k, t] - m.Q_new[i, s, k, t - 1] <= m.Q_delta_abs[
            i, s, k
        ] + (m.Q_IR_cap[i, s, k] * m.Q_new[i, s, k, t - 1])

    # ==========================================
    # IDLE CAPACITY EQUATIONS
    # ==========================================
    @m.Constraint(m.region, m.sector, m.tech_point, m.y)
    def deployment_limit(m, i, s, k, t):
        emissions = (
                            sum(m.f_NG_intern[j, i, s, t] for j in m.region if j != i)
                            + m.f_NG_import[i, s, t]
                    ) * m.psi[t]

        return (m.Q_cap[i, s, k, t] - m.Q_idle[i, s, k, t]) <= m.rho[s, t] * emissions

    @m.Constraint(m.region, m.sector, m.tech, m.y)
    def active_capacity_limit(m, i, s, k, t):
        return sum(m.q_CO2_cap[i, s, k, v, t] for v in m.v if v <= t) <= (m.Q_cap[i, s, k, t] - m.Q_idle[i, s, k, t])

    # ==========================================
    # SINK CONSTRAINTS & STAGED DEVELOPMENT
    # ==========================================
    @m.Constraint(m.region, m.sink_blocks)
    def sink_explore(m, i, b):
        if dict_cap.get((i, b), 0) == 0:
            return sum(m.sink_bdv[i, b, t] for t in m.y) == 0
        return sum(m.sink_bdv[i, b, t] for t in m.y) <= 1

    @m.Constraint(m.region, m.sink_blocks, m.y)
    def sink_stage_sequence(m, i, b, t):
        current_stage = dict_stage.get((i, b), 1)

        if current_stage <= 1:
            return pyo.Constraint.Skip

        parent_id = dict_parent.get((i, b))

        prereq_block = None
        for b_other in m.sink_blocks:
            if dict_parent.get((i, b_other)) == parent_id and dict_stage.get((i, b_other)) == current_stage - 1:
                prereq_block = b_other
                break

        if not prereq_block:
            return pyo.Constraint.Skip

        current_dev = sum(m.sink_bdv[i, b, tau] for tau in m.y if tau <= t)
        prereq_dev = sum(m.sink_bdv[i, prereq_block, tau] for tau in m.y if tau <= t)

        return current_dev <= prereq_dev

    @m.Constraint(m.region, m.sink_blocks, m.y)
    def eq_sink_active_cap(m, i, b, t):
        delay = dict_delay.get((i, b), 0) + 1
        inj_rate = dict_inj_rate.get((i, b), 0.0)

        valid_years = [tau for tau in m.y if tau <= t - delay]

        if not valid_years or inj_rate == 0:
            return m.sink_active_cap[i, b, t] == 0

        return (
                m.sink_active_cap[i, b, t]
                == sum(m.sink_bdv[i, b, tau] for tau in valid_years) * inj_rate
        )

    @m.Constraint(m.region, m.sink_blocks, m.y)
    def sink_inj_limit(m, i, b, t):
        if dict_inj_rate.get((i, b), 0) == 0:
            return m.q_CO2_inj[i, b, t] == 0
        return m.q_CO2_inj[i, b, t] <= m.sink_active_cap[i, b, t]

    @m.Constraint(m.region, m.sink_blocks, m.y)
    def sink_cum_cap(m, i, b, t):
        cap = dict_cap.get((i, b), 0)
        if cap == 0:
            return pyo.Constraint.Skip

        valid_years = [tau for tau in m.y if tau <= t]
        return (
                sum(m.q_CO2_inj[i, b, tau] for tau in valid_years) <= cap
        )

    return m
