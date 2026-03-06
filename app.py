import streamlit as st
import pandas as pd
import altair as alt
import torch

from eviot.runners.single_query import run_context_construction, CONFIG
from eviot.query.decompose_spacy import extract_phrases


st.set_page_config(
    page_title="MakeSense – Context Construction",
    page_icon="🧠",
    layout="centered"
)

st.title("MakeSense")
st.caption("Optimal Transport based semantic coverage retrieval")


query = st.text_area(
    "Query",
    height=120,
    placeholder="Enter a query..."
)

candidates_input = st.text_area(
    "Candidate Sentences",
    height=260,
    placeholder="Paste candidate sentences (one per line)"
)

mode_ui = st.selectbox(
    "Retrieval Mode",
    ["Adaptive", "Fixed", "Temporal"]
)

CONFIG["mode"] = mode_ui.lower()


# ------------------------------------------------
# Fixed-k control
# ------------------------------------------------

if CONFIG["mode"] == "fixed":

    k_fixed = st.slider(
        "Fixed Context Size (k)",
        min_value=1,
        max_value=15,
        value=CONFIG.get("k_fixed", 5),
        step=1
    )

    CONFIG["k_fixed"] = k_fixed


run = st.button("Build Context")


if run:

    if not query or not candidates_input:
        st.warning("Please provide query and candidates.")
        st.stop()

    candidate_texts = [
        x.strip()
        for x in candidates_input.split("\n")
        if x.strip()
    ]

    with st.spinner("Constructing semantic context..."):
        output = run_context_construction(query, candidate_texts)

    st.divider()
    st.subheader("Selected Context")


    if output["mode"] == "temporal":

        states = output["states"]

        for t, state in enumerate(states, 1):

            with st.expander(f"State {t}", expanded=True):

                st.write(f"Coverage OT cost: **{state['coverage_cost']:.4f}**")
                st.write(f"Temporal OT cost: **{state['temporal_cost']:.4f}**")
                st.write(f"Joint objective: **{state['objective']:.4f}**")

                for i, s in enumerate(state["context"], 1):
                    st.markdown(f"**[{i}]** {s['text']}")

        st.divider()
        st.subheader("Temporal OT Dynamics")

        coverage = [s["coverage_cost"] for s in states]
        temporal = [s["temporal_cost"] for s in states]
        objective = [s["objective"] for s in states]

        df = pd.DataFrame({
            "State": list(range(1, len(states)+1)),
            "Coverage Cost": coverage,
            "Temporal Cost": temporal,
            "Joint Objective": objective
        })

        df_melt = df.melt("State")

        chart = alt.Chart(df_melt).mark_line(point=True).encode(
            x=alt.X("State:Q", title="Temporal Step"),
            y=alt.Y("value:Q", title="Cost"),
            color="variable:N"
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Coverage cost measures semantic query coverage while "
            "temporal cost enforces consistency between context states."
        )

    else:

        context = output["context"]
        cost_curve = output["cost_curve"]

        for i, s in enumerate(context, 1):
            st.markdown(f"**[{i}]** {s['text']}")

        st.divider()

        # ------------------------------------------------
        # Semantic Coverage Curve
        # ------------------------------------------------

        st.subheader("Semantic Coverage Gain")

        coverage = [cost_curve[0] - c for c in cost_curve]

        df = pd.DataFrame({
            "Context Size": list(range(1, len(cost_curve)+1)),
            "Coverage Gain": coverage
        })

        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("Context Size:Q", title="Context Size"),
            y=alt.Y("Coverage Gain:Q", title="Semantic Coverage Gain")
        )

        st.altair_chart(chart, use_container_width=True)

        if CONFIG["mode"] == "fixed":
            st.caption(f"Fixed OT selection with k = {CONFIG['k_fixed']}")
        else:
            st.caption(
                "Coverage gain shows diminishing returns as more evidence is added."
            )

        if CONFIG["mode"] == "adaptive":

            st.divider()
            st.subheader("Adaptive Stopping Mechanism")

            epsilon = CONFIG["epsilon"]
            patience = CONFIG["patience"]

            patience_counter = 0

            for i in range(1, len(cost_curve)):

                gain = cost_curve[i-1] - cost_curve[i]

                if gain < epsilon:
                    patience_counter += 1
                    status = "Below ε"
                else:
                    patience_counter = 0
                    status = "Continue"

                st.write(
                    f"Step {i+1}: ΔOT = {gain:.5f} | ε = {epsilon} | {status}"
                )

                progress = min(1.0, max(0, gain) / epsilon)
                st.progress(progress)

                if patience_counter >= patience:
                    st.error(
                        f"Stopping triggered (patience = {patience})"
                    )
                    break


        # ------------------------------------------------
        # Greedy selection trace
        # ------------------------------------------------

        st.divider()
        st.subheader("Greedy Selection Steps")

        for i, cost in enumerate(cost_curve):

            if i == 0:
                st.write(
                    f"Step {i+1}: Initial selection → OT cost = {cost:.4f}"
                )
            else:
                gain = cost_curve[i-1] - cost
                st.write(
                    f"Step {i+1}: marginal gain = {gain:.4f} → OT cost = {cost:.4f}"
                )

        st.success(
            f"Final OT Cost: {cost_curve[-1]:.4f}"
        )

        if CONFIG["mode"] in ["adaptive", "fixed"]:

            st.divider()
            st.subheader("Query–Evidence Alignment Heatmap")

            phrases, q_embs = extract_phrases(query)

            sent_embs = torch.stack([s["emb"] for s in context])

            sim_matrix = torch.mm(q_embs, sent_embs.T).cpu().numpy()

            heat_df = pd.DataFrame(
                sim_matrix,
                index=phrases,
                columns=[f"S{i+1}" for i in range(len(context))]
            )

            heatmap = alt.Chart(
                heat_df.reset_index().melt("index")
            ).mark_rect().encode(
                x=alt.X("variable:N", title="Evidence Sentence"),
                y=alt.Y("index:N", title="Query Phrase"),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["index", "variable", "value"]
            )

            st.altair_chart(heatmap, use_container_width=True)

            st.caption(
                "Heatmap shows semantic alignment between query phrases and selected evidence sentences."
            )