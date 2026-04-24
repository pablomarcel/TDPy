from __future__ import annotations

from interpreter import InterpretConfig, interpret_text


def test_reversible_like():
    txt = """
    title: Klein & Nellis Example 5.2-1

    # givens
    g = 9.80665
    P_atm = 100000.0
    A_c = 0.18
    P_1 = 450000.0
    V_1 = 0.45

    # equations (messy order)
    W_rev = P_1*V_1*log(V_2/V_1)
    P_atm*A_c + m_p*g = P_2*A_c
    V_2 - 2*V_1 = 0
    P_2*V_2 = P_1*V_1
    W_irrev = (P_atm*(V_2-V_1) + m_p*g*(V_2-V_1)/A_c)
    """

    res = interpret_text(txt, cfg=InterpretConfig(tol=1e-6, max_iter=50))
    assert res.ok
    assert res.spec["problem_type"] == "equations"
    assert "constants" in res.spec
    assert "equations" in res.spec
    assert "variables" in res.spec
