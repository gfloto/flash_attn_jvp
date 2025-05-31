import functools
import torch


def finite_element_jvp(fn, x, z, e=1e-8):
    return (fn(x + e*z) - fn(x)) / e


def flash_attention_jvp(x, z, Wq, Wk, Wv, Br=4, Bc=3):
    n, d = x.shape
    assert n % Br == 0 and n % Bc == 0
    y, g = torch.zeros_like(x), torch.zeros_like(x)
    m, l, mu = -float("inf") * torch.ones(n), torch.zeros(n), torch.zeros(n)

    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    q_t, k_t, v_t = z @ Wq.T, z @ Wk.T, z @ Wv.T
    for j in range(n // Bc):
        k_j, v_j = k[j*Bc : (j+1)*Bc], v[j*Bc : (j+1)*Bc]
        k_t_j, v_t_j = k_t[j*Bc : (j+1)*Bc], v_t[j*Bc : (j+1)*Bc]

        for i in range(n // Br):
            y_i = y[i*Br : (i+1)*Br]
            q_i = q[i*Br : (i+1)*Br]
            q_t_i = q_t[i*Br : (i+1)*Br]
            l_i = l[i*Br : (i+1)*Br]
            m_i = m[i*Br : (i+1)*Br]
            mu_i = mu[i*Br : (i+1)*Br]
            g_i = g[i*Br : (i+1)*Br]

            s_ij = (q_i @ k_j.T) / d ** 0.5
            s_t_ij = (q_i @ k_t_j.T + q_t_i @ k_j.T) / d ** 0.5

            m_tilde_ij = s_ij.max(axis=-1)[0]
            p_tilde_ij = (s_ij - m_tilde_ij[:, None]).exp()
            l_tilde_ij = p_tilde_ij.sum(axis=-1)
            mu_tilde_j = (p_tilde_ij * s_t_ij).sum(axis=-1)
            
            m_new = torch.max(m_i, m_tilde_ij)[0]
            l_new = (m_i - m_new).exp() * l_i + (m_tilde_ij - m_new).exp() * l_tilde_ij
            
            mu_a = l_i * (m_i - m_new).exp() * mu_i
            mu_b = (m_tilde_ij - m_new).exp() * mu_tilde_j
            mu[i*Br : (i+1)*Br] = (mu_a + mu_b) / l_new

            y_a = l_i[:, None] * (m_i - m_new).exp()[:, None] * y_i 
            y_b = (m_tilde_ij - m_new).exp()[:, None] * p_tilde_ij @ v_j
            y[i*Br : (i+1)*Br] = (y_a + y_b) / l_new[:, None]

            g_a = l_i[:, None] * (m_i - m_new).exp()[:, None] * g_i
            g_b1 = p_tilde_ij @ v_t_j
            g_b2 = (p_tilde_ij * s_t_ij) @ v_j
            g_b = (m_tilde_ij - m_new).exp()[:, None] * (g_b1 + g_b2) 
            g[i*Br : (i+1)*Br] = (g_a + g_b) / l_new[:, None]

            l[i*Br : (i+1)*Br] = l_new
            m[i*Br : (i+1)*Br] = m_new

    return {"primal": y, "tangent": (g - mu[:, None] * y)}


if __name__ == "__main__":
    N, D = 12, 3
    x, z = torch.randn(2, N, D, dtype=torch.float64)
    Wq, Wk, Wv = torch.randn(3, D, D, dtype=torch.float64)
    
    assert torch.allclose(
        flash_attention_jvp(x, z, Wq, Wk, Wv)["primal"],
        torch.nn.functional.scaled_dot_product_attention(x @ Wq.T, x @ Wk.T, x @ Wv.T),
        rtol=0.0,
        atol=1e-2,
    )
    
    partial_attention = functools.partial(
        lambda x_: torch.nn.functional.scaled_dot_product_attention(
            x_ @ Wq.T, x_ @ Wk.T, x_ @ Wv.T
        )
    )
    assert torch.allclose(
        flash_attention_jvp(x, z, Wq, Wk, Wv)["tangent"],
        finite_element_jvp(partial_attention, x, z),
        rtol=0.0,
        atol=1e-2,
    )

