import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.code_loader import show_source, run_it_yourself
st.set_page_config(page_title="Learning Rate Schedules | llms102", layout="wide")

# ─── Hero ────────────────────────────────────────────────────────────────────

st.title("Learning Rate Schedules")
st.caption("Broad learning first, fine learning later")

st.info("This chapter is incomplete, but I am leaving here what I have written for anyone who might to reference it. Honestly, there isn't much to understand here other than the fact that we have coarse and fine updates and we want our LR to to work like that as well")
with st.expander("Draft chapter", expanded=False):
    st.markdown("""
    When training a transformer network from scratch, we want to set the model up for success, not failure. For instance, if you were painting on a large canvas, you wouldn't start with the intricate details of the hair. You would first paint broadstrokes on the whole canvas, laying out the background, the border, etc. and then start filling in minutiae. 
                
    Likewise, we want our network to first learn the broadstrokes of the task (high learning rate) and then focus on detail (low rate). Le's explore how these learning rate schedules can be set. 
    """)

    def constant_schedule(lr, n=100):
        return np.full(n, lr)

    def step_decay(init_lr, gamma, interval, n=100):
        return np.array([init_lr * (gamma ** (i // interval)) for i in range(n)])

    def exp_decay(init_lr, gamma, n=100):
        return np.array([init_lr * (gamma ** i) for i in range(n)])

    def cosine_schedule(max_lr, min_lr, t_max, n=100):
        return np.array([
            min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * (i % t_max) / t_max))
            for i in range(n)
        ])

    def warmup_cosine(peak_lr, warmup_steps, total_steps, n=100):
        lrs = []
        for i in range(n):
            step = i * total_steps / n
            if step < warmup_steps:
                lrs.append(peak_lr * step / warmup_steps)
            else:
                prog = (step - warmup_steps) / (total_steps - warmup_steps)
                lrs.append(peak_lr * 0.5 * (1 + np.cos(np.pi * prog)))
        return np.array(lrs)

    def simulate_loss(lr_schedule, seed=42):
        rng = np.random.default_rng(seed)
        loss, losses = 2.5, []
        for lr in lr_schedule:
            loss = max(0.05, loss - lr * (loss + 0.1) * 0.4 + rng.random() * 0.02)
            losses.append(round(loss, 3))
        return losses

    # ── Plotly helpers ─────────────────────────────────────────────────────────────

    COLORS = {
        "constant":     "#378ADD",
        "step":         "#D85A30",
        "exp":          "#1D9E75",
        "cosine":       "#7F77DD",
        "warmup":       "#7F77DD",
        "warmup_decay": "#7F77DD",
    }

    def base_fig(y_title="Learning rate", height=280):
        fig = go.Figure()
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            showlegend=False,
            xaxis=dict(title="Step", gridcolor="rgba(136,135,128,0.15)", zeroline=False),
            yaxis=dict(title=y_title, gridcolor="rgba(136,135,128,0.15)", zeroline=False),
        )
        return fig

    # ── Section 1: What is the learning rate? ─────────────────────────────────────

    st.divider()
    st.subheader("What is the learning rate?")

    st.markdown("""
    When performing gradient descent on a network, we find the gradient of the output compared to the ground truth. Based on this, we apply an update to the parameters. The learning rate modulates how much of this update will be applied. 
                
    We want the learning rate of the optimizer to push the parameters towards values that lead to a lower loss. Additionally, during the initial stages, we want larger updates since there is a lot of correction to be done, while during later stages we want to slow down some of this learning so that gradient updates do not push the parameters towards the wrong direction. It could also very well push the model into a different local minima which may in fact not be the global minima.
                
    There is also the issue of encountering unseen data. Let's say your model is trained largely on grade 5 english textbooks, and it suddenly encounters a bad batch of data, which is in fact grade 5 of a master's degree course. The content would be considerably more complex, and modelling it would be harder. In such a case, we do not want a catastrophically high LR that pushes the parameters too far away from its original learning (often an issue during finetuning on custom datasets). 

    Then there is the matter of convergence - large gradient updates applied early on allow us to find the approximate region in which we might come across the minima, while later softer updates will allow us to traverse the loss surface more carefully, assisting in accuracy.             

    Below, we present a relatively smooth two dimensional parabolic loss surface. Try moving the slider between different values and notice how the updates occur at different rates.  
    """)

    lr_val = st.slider("Learning rate", min_value=0.001, max_value=0.8,
                    value=0.03, step=0.001, format="%.3f", key="lr_landscape")

    x = np.linspace(-3, 3, 200)
    loss_surface = x ** 2 + 0.3
    x0 = 2.0
    grad = 2 * x0
    x1 = x0 - lr_val * grad

    fig = base_fig(y_title="Loss", height=300)
    fig.add_trace(go.Scatter(x=x, y=loss_surface, mode="lines",
                            line=dict(color=COLORS["constant"], width=2), name="Loss surface"))
    fig.add_trace(go.Scatter(x=[x0, x1], y=[x0**2 + 0.3, x1**2 + 0.3],
                            mode="markers+lines",
                            marker=dict(color=COLORS["step"], size=10),
                            line=dict(color=COLORS["step"], width=1.5, dash="dot"),
                            name="Gradient step"))
    fig.update_xaxes(title="Parameter value")
    st.plotly_chart(fig, use_container_width=True)

    if lr_val > 0.07:
        st.warning("High LR. Loss may diverge.")
    elif lr_val < 0.01:
        st.info("Very small LR — stable but slow. May get stuck in shallow minima.")
    else:
        st.success("Balanced LR — steady progress toward the minimum.")

    # ── Section 2: Constant LR ────────────────────────────────────────────────────

    st.divider()
    st.subheader("Constant learning rate")

    st.markdown("""
    As a baseline, let's just hold the learning rate constant. The obvious issue with this is that the gap in updates we require at step 1 and step 100k is very very large - we should not be applying the same rates to both. If we lower the rate so it remains stable, we need more steps for convergence. If we increase the LR for convergence, we might see instability in later steps owing to large updates. 
    """)

    st.subheader("Step decay & exponential decay")
    st.caption("Drop it like a step function, or smoothly")

    st.markdown("""
    Step decay basically reduces the LR by a factor of decay rate at some fixed interval. This means we first reduce LR a lot, then slowly reduce the rate of LR update. You can see the graph below for an idea of what it looks like. 

    This does present some issues, however - how do we decide what the decay rate and step interval size should be? These would then have to come from empirical investigation. There is also the issue of a sudden change in the update size. What if we could have something smoother? 
                
    The simplest answer to the smoothing, is to apply an exponentially decaying function to the LR. Both of these were commonly used, but are not anymore in favour of other schedulers. 
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        init_lr   = st.slider("Initial LR", 0.01, 0.1, 0.06, 0.005, format="%.3f", key="decay_init")
    with col2:
        gamma     = st.slider("Decay rate γ", 0.5, 0.99, 0.85, 0.01, format="%.2f", key="decay_gamma")
    with col3:
        interval  = st.slider("Step interval (epochs)", 1, 20, 5, key="step_interval")

    step_sched = step_decay(init_lr, gamma, interval)
    exp_sched  = exp_decay(init_lr, gamma)

    fig = base_fig()
    fig.add_trace(go.Scatter(y=step_sched, mode="lines", name="Step decay",
                            line=dict(color=COLORS["step"], width=2)))
    fig.add_trace(go.Scatter(y=exp_sched, mode="lines", name="Exp decay",
                            line=dict(color=COLORS["exp"], width=2, dash="dash")))
    fig.update_layout(showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── Section 4: Cosine annealing ───────────────────────────────────────────────

    st.divider()
    st.subheader("4. Cosine annealing")
    st.caption("Smooth, principled, and everywhere in modern ML")

    st.markdown("""
    *✏️ Why cosine? The smooth shape matches the geometry of loss landscapes better
    than linear decay. Explain T_max, η_min, and optional restarts (SGDR).*
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cos_max      = st.slider("Max LR", 0.01, 0.1, 0.06, 0.005, format="%.3f", key="cos_max")
    with col2:
        cos_min      = st.slider("Min LR", 0.0, 0.02, 0.002, 0.001, format="%.3f", key="cos_min")
    with col3:
        t_max        = st.slider("T_max (steps)", 10, 100, 50, key="cos_tmax")
    with col4:
        restarts     = st.slider("Restarts", 1, 5, 1, key="cos_restarts")

    n_steps = t_max * restarts
    cos_sched = cosine_schedule(cos_max, cos_min, t_max, n=n_steps)

    fig = base_fig()
    fig.add_trace(go.Scatter(y=cos_sched, mode="lines",
                            line=dict(color=COLORS["cosine"], width=2)))
    if restarts == 1:
        st.caption("Standard cosine annealing — no restarts.")
    else:
        st.caption(f"SGDR with {restarts} warm restarts.")
    st.plotly_chart(fig, use_container_width=True)

    # ── Section 5: Warmup + cosine (transformer schedule) ─────────────────────────

    st.divider()
    st.subheader("5. Warmup + cosine decay")
    st.caption("The standard for training transformers from scratch")

    st.markdown("""
    *✏️ The star of the show. Explain why transformers need warmup (unstable gradients,
    layer norm sensitivity). Reference Vaswani et al. Show the Noam schedule formula.*
    """)

    st.markdown("""
    *✏️ Add a code snippet here — e.g. `get_cosine_schedule_with_warmup` from
    🤗 Transformers, or a custom PyTorch lambda scheduler.*
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        peak_lr      = st.slider("Peak LR", 1e-4, 1e-3, 3e-4, 1e-5, format="%.0e", key="t_peak")
    with col2:
        warmup_steps = st.slider("Warmup steps", 100, 10000, 4000, 100, key="t_warmup")
    with col3:
        total_steps  = st.slider("Total steps", 5000, 100000, 40000, 1000, key="t_total")

    wu_sched = warmup_cosine(peak_lr, warmup_steps, total_steps)
    split    = int(warmup_steps / total_steps * 100)

    fig = base_fig(height=300)
    fig.add_trace(go.Scatter(y=wu_sched[:split+1], mode="lines", name="Warmup",
                            line=dict(color=COLORS["warmup"], width=2)))
    fig.add_trace(go.Scatter(x=list(range(split, 100)), y=wu_sched[split:],
                            mode="lines", name="Cosine decay",
                            line=dict(color=COLORS["warmup_decay"], width=2, dash="dash")))
    fig.update_layout(showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    st.plotly_chart(fig, use_container_width=True)

    warmup_pct = warmup_steps / total_steps * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("Warmup %", f"{warmup_pct:.1f}%")
    m2.metric("Peak LR",  f"{peak_lr:.1e}")
    m3.metric("Total steps", f"{total_steps:,}")

    if warmup_steps < 500:
        st.warning("Warmup may be too short for a transformer — consider at least 1000–4000 steps.")
    elif warmup_pct > 20:
        st.warning("Warmup is a large fraction of training — consider reducing it.")
    else:
        st.success("Warmup looks reasonable.")

    # ── Section 6: Side-by-side comparison ────────────────────────────────────────

    st.divider()
    st.subheader("6. All schedules compared")
    st.caption("Same axis, same training run length")

    st.markdown("""
    *✏️ Closing summary: practical guidance — when to use each schedule, typical
    hyperparameter ranges, and how to diagnose problems (loss spikes → LR too high,
    plateau → LR too low or warmup too short).*
    """)

    N = 100
    comp_const = constant_schedule(0.03, N)
    comp_step  = step_decay(0.06, 0.88, 5, N)
    comp_cos   = cosine_schedule(0.06, 0.002, 50, N)
    comp_wu    = warmup_cosine(0.03, 8, N, N)

    fig = base_fig(height=340)
    fig.add_trace(go.Scatter(y=comp_const, mode="lines", name="Constant",
                            line=dict(color=COLORS["constant"], width=2)))
    fig.add_trace(go.Scatter(y=comp_step,  mode="lines", name="Step decay",
                            line=dict(color=COLORS["step"],     width=2)))
    fig.add_trace(go.Scatter(y=comp_cos,   mode="lines", name="Cosine",
                            line=dict(color=COLORS["exp"],      width=2)))
    fig.add_trace(go.Scatter(y=comp_wu,    mode="lines", name="Warmup + cosine",
                            line=dict(color=COLORS["cosine"],   width=2, dash="dot")))
    fig.update_layout(showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0))
    st.plotly_chart(fig, use_container_width=True)