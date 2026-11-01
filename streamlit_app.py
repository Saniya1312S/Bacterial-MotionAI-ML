"""
BacMotionAI demo app - uses the REAL trained model (models/best_model.joblib).

Run: streamlit run streamlit_app.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.simulation import simulate_bacterium, CHEMICAL_LABELS
from src.inference import load_best_model, predict_trajectory

st.set_page_config(page_title="BacMotionAI", page_icon="🦠", layout="wide")
st.title("🦠 BacMotionAI - Which chemical is this bacterium swimming in?")
st.write(
    "Simulate one E. coli trajectory, then let the trained ML model guess the chemical condition "
    "from the path alone. The model never sees the true label."
)


@st.cache_resource
def get_model():
    return load_best_model()


model = get_model()

with st.sidebar:
    st.header("Simulation settings")
    true_chem = st.selectbox("True chemical (hidden from the model)", CHEMICAL_LABELS, index=1)
    width = st.select_slider("Channel width (µm)", options=[10, 20, 30, 50, 75, 100], value=30)
    hydro = st.checkbox("Wall hydrodynamic interaction", value=True)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=7, step=1)

xs, ys, _ = simulate_bacterium(width, hydro, true_chem, seed=int(seed))
pred, probs, feats = predict_trajectory(model, xs, ys, width)

left, right = st.columns(2)

with left:
    st.subheader("Trajectory (6 seconds)")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs - xs[0], ys, lw=1.2, color="#2563EB")
    ax.scatter([0], [ys[0]], color="green", zorder=3, label="start")
    ax.scatter([xs[-1] - xs[0]], [ys[-1]], color="red", zorder=3, label="end")
    ax.axhline(0, color="k", lw=2)
    ax.axhline(width, color="k", lw=2)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.legend(loc="upper right")
    st.pyplot(fig)

with right:
    st.subheader("Model prediction")
    ok = pred == true_chem
    (st.success if ok else st.error)(
        f"Predicted: **{pred}**   |   True: **{true_chem}**   {'✅' if ok else '❌'}"
    )
    prob_df = pd.DataFrame({"probability": probs}).sort_values("probability", ascending=False)
    st.bar_chart(prob_df)

st.subheader("Features extracted from this trajectory")
st.dataframe(pd.DataFrame([feats]).round(3), hide_index=True)

st.caption(
    "Data is simulated, so accuracy here reflects how separable the simulated conditions are, "
    "not real-lab performance. Try several seeds - the model is right about 87% of the time, not always."
)
